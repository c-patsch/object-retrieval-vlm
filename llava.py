import torch
import cv2
from transformers import BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration
from torchvision import transforms

class LlavaModel:
    def __init__(self):
        # Initialize the processor and model
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        
        # Bits and Bytes config for quantization
        self.bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        # Load the model with the configuration
        self.model_vlm = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        #self.model_vlm.eval()  # Set model to evaluation mode


    def preprocess_image(self, frame):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        return transform(frame).unsqueeze(0)  # Add batch dimension


    def generate_caption(self, frame, object_label):
        """
        Generate caption for the given frame and object label.
        
        :param frame: Image frame to process.
        :param object_label: Label for the object in the image.
        :return: Generated caption for the object.
        """
        print(object_label)

        #torch.cuda.empty_cache()


        resized_frame = cv2.resize(frame, (336, 336))  # Ensure resizing to correct dimensions
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        #resized_frame = self.preprocess_image(resized_frame)
        #prompt = f"<image>\n{object_label} has been observed. What is the person doing with it?"
        prompt = "Describe the image in one short sentence."
        print(f"Prompt: {prompt}")  # Debugging: Print the prompt

        # Prepare inputs for the model
        #inputs = self.processor(text=prompt, images=resized_frame, return_tensors="pt").to("cuda", torch.float16)

        if object_label == None:
            conversation = [
                        {

                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"What is shown in this image in one very short sentence ?"},
                            {"type": "image"},
                            ],
                        },
                    ]
        elif isinstance(object_label, str):

            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": f"What is shown in this image in one very short short sentence given that the person is interacting with {object_label}?"},
                    {"type": "image"},
                    ],
                },
            ]

        elif len(object_label) >= 2:

            object_label = " and ".join(object_label)

            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": f"What is shown in this image in one very short short sentence given that the person is interacting with {object_label}?"},
                    {"type": "image"},
                    ],
                },
            ]           
        else:
            object_label = object_label[0]
            
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": f"What is shown in this image in one very short sentence given that the person is interacting with {object_label}?"},
                    {"type": "image"},
                    ],
                },
            ]



        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=resized_frame, text=prompt, return_tensors="pt").to("cuda:0")

        #hi = inputs.data['pixel_values'][0][0].permute(2,0,1).cpu().numpy()
        #cv2.imwrite('hi.jpg', resized_frame)

        # Inference
        with torch.no_grad():
            output = self.model_vlm.generate(**inputs, max_new_tokens=16, temperature = 0.1, do_sample = True)

        print(f"Raw model output: {output}")  # Debugging: Print raw output

        # Decode the output from the model
        raw_output = self.processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        print(f"Decoded caption: {raw_output}")  # Debugging: Print decoded caption

        # Clean the caption by removing the object-specific prompt
        caption = raw_output.split("ASSISTANT: ")[-1]#.replace(f"{object_label} has been observed. What is the person doing with it?", "").replace("\n", "")
        
        return caption
