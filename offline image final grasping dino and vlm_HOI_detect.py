import cv2
#import mediapipe as mp
import numpy as np
import os
from collections import deque
from ultralytics import SAM
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from llava import LlavaModel

from blip2 import Blip2Model



import os
import random

def sample_indices_from_annotation(file_path, num_samples=5):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                range_part, caption = line.strip().split(':', 1)
                try:
                    start, end = map(int, range_part.strip().split('-'))
                    if end >= start:
                        sampled = random.sample(range(start + 20, end - 20 + 1), min(num_samples, end - start + 1))
                        samples.append((sampled, caption.strip()))
                except ValueError:
                    print(f"Could not parse line: {line.strip()}")
    return samples


split_nr = 1
# Path to your list of annotation filenames
with open(f"/media/constantin/b348f37b-50b5-4230-9930-608d98ffb4b2/gtea/splits/test.split{split_nr}.bundle", "r") as f:
    annotation_names = [line.strip() for line in f if line.strip()]




# loaded_array = np.load("all_gts_split1.npy")
# loaded_array2 = np.load("all_preds_split1.npy")








# Load SAM model
sam_model = SAM("mobile_sam.pt")

# MediaPipe setup
#mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)


# Load LLaVA
llava_model =  LlavaModel()

# Load DINOv2 model
device = 'cuda:0'
dtype = torch.float16
dinov2_model_name = "facebook/dinov2-large"
dinov2_model = AutoModel.from_pretrained(dinov2_model_name, torch_dtype=dtype).to(device).eval()
dinov2_processor = AutoProcessor.from_pretrained(dinov2_model_name)

# Embedding function
def extract_embeddings(image_paths):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = dinov2_processor(images=images, return_tensors="pt").to(device, dtype=dtype)
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return (embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()

# Prepare your database
img_data_base = "/media/constantin/b348f37b-50b5-4230-9930-608d98ffb4b2/gtea/object pics"
image_paths = [os.path.join(img_data_base, f) for f in os.listdir(img_data_base) if f.endswith((".jpg", ".png"))]
database_embeddings = extract_embeddings(image_paths)
neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(database_embeddings)

# Constants
INDEX_FINGER_ID = 8
THUMB_TIP_ID = 4
INDEX_FINGER_TIP_ID = 8

KERNEL = np.ones((20, 20), np.uint8)

#mp_drawing = mp.solutions.drawing_utils

# Process a single image
def process_image(image_path, detections, ind, gt):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #results = hands.process(rgb)

    image_draw = image.copy()

    caption_obj = ''
    caption = ''



    #if results.multi_hand_landmarks:
    if True:
    #if False:
        most_right_x = -1
        best_point = None

        hand_box = None
        most_right_x = -1


        
        #hand_dets = detections.get("hand_dets", [])
        obj_dets = detections

        obj_box = None
        most_right_x = -1

        # # Choose the rightmost hand box
        # for box in obj_dets:
        #     x_min, y_min, x_max, y_max = box
        #     interp_x = (x_min + x_max) // 2
        #     interp_y = (y_min + y_max) // 2

        #     if interp_x > most_right_x:
        #         most_right_x = interp_x
        #         hand_box = [int(x_min), int(y_min), int(x_max), int(y_max)]  # [x0, y0, x1, y1]

        obj_box = detections

        all_labels = []
        for ind in range(obj_box.shape[0]):

            curr_obj_dets = obj_dets[ind][:4]
            # SAM expects [x0, y0, x1, y1]
            # Pass the bounding box to SAM
            if curr_obj_dets is not None:
                result = sam_model.predict(image, bboxes=[curr_obj_dets], labels=[1], verbose=False)
            else:
                hi = 1


            #result = sam_model.predict(image, points=[point], labels=[1], verbose=False)
            mask = max(result[0].masks.data.cpu().numpy(), key=lambda m: np.sum(m))
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))

            mask = cv2.dilate(mask, KERNEL, iterations=1)



            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    mask_center = np.array([cx, cy])

            x, y, w, h = cv2.boundingRect(largest_contour)


            #x, y, w, h = cv2.boundingRect(mask)
            cropped = image[y:y+h, x:x+w]
            cutout_path = "cutout.jpg"
            cv2.imwrite(cutout_path, cropped)

            query_embedding = extract_embeddings([cutout_path])
            distances, indices = neighbors.kneighbors(query_embedding)
            label = os.path.basename(image_paths[indices[0][0]]).split('.')[0]
            all_labels.append(label)

        caption = llava_model.generate_caption(image, all_labels)
        print("Caption Obejct:", caption)

        caption_no_object = llava_model.generate_caption(image, object_label = None)
        print("Caption No Obejct:", caption_no_object)

        gt = gt
        print("GT:", gt)

        #cv2.circle(image_draw, point, 6, (255, 0, 0), -1)  # Thumb tip (blue)
        cv2.circle(image_draw, mask_center, 6, (0, 255, 255), -1)  # Mask center (yellow)


        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = (0, 255, 0)  # Green color for the mask

        # Blend the colored mask with the original frame (overlay the mask)
        image_draw = cv2.addWeighted(image_draw, 1, colored_mask, 0.5, 0)

        #cv2.imshow("Smartphone Stream Joints + Object Mask", image)
        cv2.imwrite('testing.jpg', image_draw)

    else:

        caption = llava_model.generate_caption(image, object_label = None)
        #caption = ''

    return caption_obj, caption



base_dir = "new_caps"

all_preds = []
all_gts = []

for name in annotation_names:
    new_filename = f"{name.replace('.txt', '')}_new.txt"
    full_path = os.path.join(base_dir, new_filename)

    if os.path.exists(full_path):
        print(f"\nProcessing {new_filename}:")
        results = sample_indices_from_annotation(full_path)
        for indices, caption in results:
            print(f"{caption} → sampled indices: {indices}")
    else:
        print(f"File not found: {full_path}")

    for seg in results:

        caption_obj_all = []
        caption_all = []



        detections = np.load(f"/home/constantin/Desktop/Constantin/object-retrieval-vlm/bb_detections/{name.replace('.txt', '')}_detections.npy", allow_pickle=True)#.item()

        for ind in seg[0]:
            
            if detections[ind]["obj_dets"].size > 0:

                img_path = f"/media/constantin/b348f37b-50b5-4230-9930-608d98ffb4b2/gtea/pictures/{name.replace('.txt', '')}/{ind:010d}.png"
                # Example call
                caption_obj, caption = process_image(img_path, detections[ind]["obj_dets"], ind, seg[1])

                caption_obj_all.append(caption_obj)
                caption_all.append(caption)

                hi = 1
            else:
                caption_all.append("")

        if caption_all:
            if all(x == '' for x in caption_all):
                image = cv2.imread(img_path)
                final_cap = llava_model.generate_caption(image, None)
            else:
                # before HOI eval with this
                #final_cap = caption_all[0]
                non_empty_captions = [cap for cap in caption_all if cap.strip() != ""]
                final_cap = random.choice(non_empty_captions)
                final_cap = non_empty_captions[-1]


        else:
            final_cap = caption#[0]
        
        all_preds.append(final_cap)
        all_gts.append(seg[1])


np.save(f"all_preds_split{split_nr}.npy", all_preds)
np.save(f"all_gts_split{split_nr}.npy", all_gts)
