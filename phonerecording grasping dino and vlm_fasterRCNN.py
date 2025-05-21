import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import SAM
import os




import torch
import torch.nn as nn
#from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from PIL import Image

from gtts import gTTS
from playsound import playsound


from transformers import AutoProcessor, AutoModel
import threading
from llava import LlavaModel

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
import argparse

from model.faster_rcnn.resnet import resnet
import _frozen_importlib_external

import time


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='hand_object_detector/cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="/media/constantin/b348f37b-50b5-4230-9930-608d98ffb4b2/gtea/pictures/S1_Cheese_C1")
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save results',
                      default="images_det")
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true', default= True)
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=132028, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=0, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)




participant = input('Participant name')
os.makedirs(f"wild_dataset/{participant}", exist_ok=True)


args = parse_args()

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

cfg.USE_GPU_NMS = args.cuda
np.random.seed(cfg.RNG_SEED)
# load model
model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
if not os.path.exists(model_dir):
    raise Exception('There is no input directory for loading network from ' + model_dir)
load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 
# initilize the network here.
fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
fasterRCNN.create_architecture()
checkpoint = torch.load(load_name)
fasterRCNN.load_state_dict(checkpoint['model'])

if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

print('load model successfully!')


# initilize the tensor holder here.
im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)
box_info = torch.FloatTensor(1) 

# ship to cuda
#if args.cuda > 0:
im_data = im_data.cuda()
im_info = im_info.cuda()
num_boxes = num_boxes.cuda()
gt_boxes = gt_boxes.cuda()


cfg.CUDA = True

#if args.cuda > 0:
fasterRCNN.cuda()

fasterRCNN.eval()
max_per_image = 100
thresh_hand = args.thresh_hand 
thresh_obj = args.thresh_obj
vis = args.vis

num_images = 0
FRAME_SKIP = 5  # Change as needed
frame_count = 0
all_detections = []































# Initialize the Llava model
llava_model = LlavaModel()


# Image database directory path:
img_data_base = f"wild_dataset/{participant}/database" #"captured_images_masked"

# # Initialize the TTS engine
# engine = pyttsx3.init()

# # Set properties (optional)
# engine.setProperty('rate', 100)  # Speed of speech
# engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

dtype = torch.float16
# Load DINOv2 model and processor # facebook/dinov2-base
dinov2_model_name = "facebook/dinov2-large"  # or 'dinov2-large', 'dinov2-giant'
dinov2_model = AutoModel.from_pretrained(dinov2_model_name, torch_dtype=dtype)
dinov2_processor = AutoProcessor.from_pretrained(dinov2_model_name)
device = 'cuda:0'
dinov2_model.to(device)
dinov2_model.eval()

# Custom Dataset for PIL Images
class DinoImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return Image.open(self.image_paths[idx]).convert("RGB")

# Custom collate function to avoid auto tensor conversion
def collate_fn(batch):
    return batch

# Embedding extraction function for DINOv2
def extract_embeddings(image_paths, batch_size=16):
    dataset = DinoImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []
    with torch.no_grad():
        for batch_images in dataloader:
            inputs = dinov2_processor(images=batch_images, return_tensors="pt")

            # Move each tensor in the inputs to the GPU
            inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}
            outputs = dinov2_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)  # L2 normalization
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

# Example usage
image_paths = [os.path.join(img_data_base, fname) for fname in os.listdir(img_data_base)]
database_embeddings = extract_embeddings(image_paths)

# device = 'cpu'
# dinov2_model.to(device)

# Check the shape of the embeddings
print("Database embeddings shape:", database_embeddings.shape)

# Use NearestNeighbors from scikit-learn for fast similarity search
neighbors = NearestNeighbors(n_neighbors=5, metric='euclidean')
neighbors.fit(database_embeddings)







# Load SAM model
sam_model = SAM("mobile_sam.pt")

# Constants
STATIC_FRAMES = 3
PIXEL_TOLERANCE = 30
INDEX_FINGER_ID = 8
FRAME_SKIP = 10  # Apply detection every 5 frames
KERNEL = np.ones((25, 25), np.uint8)

THUMB_TIP_ID = 4
DISTANCE_TOLERANCE = 30  # pixels
distance_buffer = deque(maxlen=STATIC_FRAMES)

def is_distance_constant():
    if len(distance_buffer) < STATIC_FRAMES:
        return False
    diffs = np.abs(np.diff(distance_buffer))
    return np.all(diffs < DISTANCE_TOLERANCE)

# # Setup buffer to check for static index finger
# index_finger_buffer = deque(maxlen=STATIC_FRAMES)

# # Helper functions
# def is_static():
#     if len(index_finger_buffer) < STATIC_FRAMES:
#         return False
#     pos = np.array(index_finger_buffer)  # Corrected variable name here
#     return (pos.max(axis=0) - pos.min(axis=0) <= PIXEL_TOLERANCE).all()

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start webcam or video stream
cap = cv2.VideoCapture("http://10.181.123.162:4747/video")

frame_count = 0  # To keep track of frame numbers

# Define kernel for dilation
kernel = np.ones((25, 25), np.uint8)  # Large kernel to dilate the mask

# Create the 'cutout' directory if it doesn't exist
if not os.path.exists('cutout'):
    os.makedirs('cutout')

# This will hold masks for each frame
saved_masks = []

cnter = 0
blocked = False
mask_cnt = 0

prev_lab = ''

audio_cnt = 0

# Keep this globally or in an object
ema_center = None
alpha = 0.5  # smoothing factor (0.0 = no update, 1.0 = instant update)



# Define video codec and output file #########################################
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG'
output_path = 'output_video.mp4'
fps = 6.0
frame_size = (640, 480)  # Match your frame_resized dimensions
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
#########################################################

last_caption_time = 0
obj_det_cnt = 0
cut_out_cnt = 0
times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Increment frame count

    if frame_count % int(FRAME_SKIP*10) == 0:
        ema_center = None
    # Only apply hand detection on every `frame_skip`-th frame
    if frame_count % FRAME_SKIP == 0:
        # Resize for performance and consistency
        frame_resized = frame #cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)















        init_time = time.time()


        im = np.array(frame)
        blobs, im_scales = _get_image_blob(im)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_()

            rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        contact_vector = loss_list[0][0]
        offset_vector = loss_list[1][0].detach()
        lr_vector = loss_list[2][0].detach()
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
        lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                scale = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
                shift = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                #if args.cuda > 0:
                    
                scale = scale.cuda()
                shift = shift.cuda()

                box_deltas = box_deltas.view(-1, 4) * scale + shift
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        obj_dets, hand_dets = None, None
        for j in range(1, len(pascal_classes)):
            if pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)
            else:
                continue

            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if pascal_classes[j] == 'targetobject':
                    obj_dets = cls_dets.cpu().numpy()
                elif pascal_classes[j] == 'hand':
                    hand_dets = cls_dets.cpu().numpy()







        # SAM expects [x0, y0, x1, y1]
        # Pass the bounding box to SAM
    
        if obj_dets is not None:
            curr_obj_dets = obj_dets[0][:4]

            result = sam_model.predict(frame_resized, bboxes=[curr_obj_dets], labels=[1], verbose=False)
            # Extract center of the largest mask
            #largest_mask = result[0].masks.data[0].cpu().numpy()
            masks = result[0].masks.data.cpu().numpy()
            largest_mask = max(masks, key=lambda m: np.sum(m))  # Largest mask by pixel count

            mask_resized = cv2.resize(largest_mask.astype(np.uint8), (frame_resized.shape[1], frame_resized.shape[0]))
            dilated = cv2.dilate(mask_resized, KERNEL, iterations=1)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    mask_center = np.array([cx, cy])


                    # EMA Average:
                    if ema_center is None:
                        ema_center = mask_center
                    else:
                        ema_center = alpha * mask_center + (1 - alpha) * ema_center

                    mask_center = ema_center.astype(int) #tuple([int(p) for p in ema_center])#.astype(int))
                    ############################################

                    obj_det_cnt += 1

                    if time.time() - last_caption_time >= 8 and obj_det_cnt > 5:
                        obj_det_cnt = 0
                        # generate caption
                        last_caption_time = time.time()


                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Crop the image to the bounding box of the mask
                        frame_cropped = frame_resized[y:y+h, x:x+w]
                    
                        # Now enter your SAM cropping and saving logic here...
                        print("Stable gesture detected, saving cutout...")
                        # [Insert cutout saving & voice feedback logic]

                        cutout_filename = f"cutout/{cnter}.jpg"
                        cnter += 1
                        cutout = frame_cropped  # Use cropped frame as the cutout
                        cv2.imwrite(cutout_filename, cutout)
                        blocked = True
                        mask_cnt = 0
                        saved_masks = []


                        # Load and extract embedding for the query image (cutout)
                        query_image_path = cutout_filename
                        query_embedding = extract_embeddings([query_image_path])[0].reshape(1, -1)

                        # Perform the search
                        distances, indices = neighbors.kneighbors(query_embedding)

                        most_likely = indices[0][0]
                        #output = f"You are grasping {os.path.basename(image_paths[most_likely]).replace('.jpg','').replace('.png','')}."
                        
                        label = os.path.basename(image_paths[most_likely]).replace('.jpg','').replace('.png','')  # Ensure you are dynamically setting the label, not leaving it empty
                        
                        torch.cuda.empty_cache()



                        caption_noobject = llava_model.generate_caption(frame, None)

                        #image = cv2.imread('hi.jpg')                                       
                        caption = llava_model.generate_caption(frame, label)
                        # # print("Caption:", caption)  # Debugging: Print the final caption
                        # # Read the image

                        # # vlm_thread = VLMThread(image, label)
                        # # vlm_thread.run()

                        # # # # Speak the text
                        # if caption != '' and os.path.basename(image_paths[most_likely]).replace('.jpg','').replace('.png','') != prev_lab:
                        #     # engine.say(output)
                        #     # engine.runAndWait()
                        #     # Generate TTS
                        tts = gTTS(caption, lang='en')
                        tts.save(f"output{audio_cnt}.mp3")

                        # # Play audio
                        playsound(f"output{audio_cnt}.mp3")
                        audio_cnt += 1

                        prev_lab = os.path.basename(image_paths[most_likely]).replace('.jpg','').replace('.png','')

                        print(time.time() - init_time)

                        times.append(time.time() - init_time)

                        print(np.average(times))
                        with open(f"wild_dataset/{participant}/{cut_out_cnt}_noobject.txt", "w") as f:
                            f.write(f"{caption_noobject}\n")


                        with open(f"wild_dataset/{participant}/{cut_out_cnt}.txt", "w") as f:
                            f.write(f"{caption}\n")


                        frame_study = frame_resized.copy()
                        colored_mask = np.zeros_like(frame_study)
                        colored_mask[dilated == 1] = (0, 255, 0)  # Green color for the mask

                        # Blend the colored mask with the original frame (overlay the mask)
                        frame_study = cv2.addWeighted(frame_study, 1, colored_mask, 0.5, 0)
                        cv2.circle(frame_study, mask_center, 6, (0, 255, 255), -1)  # Mask center (yellow)
                        
                        cv2.imwrite(f'wild_dataset/{participant}/{cut_out_cnt}.jpg', frame_study)
                        cut_out_cnt += 1
                    # Draw for visualization
                    #cv2.circle(frame_resized, thumb_tip_pos, 6, (255, 0, 0), -1)  # Thumb tip (blue)
                    cv2.circle(frame_resized, mask_center, 6, (0, 255, 255), -1)  # Mask center (yellow)


                # colored_mask = np.zeros_like(frame_resized)
                # colored_mask[dilated == 1] = (0, 255, 0)  # Green color for the mask

                # # Blend the colored mask with the original frame (overlay the mask)
                # frame_resized = cv2.addWeighted(frame_resized, 1, colored_mask, 0.5, 0)

         



        else:
            blocked = False
            saved_masks = []
            obj_det_cnt = 0
            
        # # # Show the frame with the overlaid mask
        #cv2.imshow("Smartphone Stream Joints + Object Mask", frame_resized)
        # # Save the frame to video
        # out.write(frame_resized)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
