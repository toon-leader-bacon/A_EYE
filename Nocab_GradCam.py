from typing import List
import PIL.Image
import clip
from clip.model import CLIP
import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import transformers
import transformers.models
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import scale_cam_image
import time


def reshape_transform(tensor: torch.Tensor, height=16, width=16):
    # Taken from example
    blab = tensor[:, 1:, :]  # [1, 256, 1024]   [1, 196, 768]
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class GradCAM:
    def __init__(self):
        # Register forward hook to capture feature maps
        self.activations = []
        self.gradients_array = []

        self.feature_maps: torch.Tensor = None  # the outputs of the target layer in the forward pass
        self.gradients: torch.Tensor = None  # the gradients of the target layer in the backwards pass

        self.device = "cpu"

        # model_name = "openai/clip-vit-base-patch16"
        # square = 14

        model_name = "openai/clip-vit-large-patch14"
        square = 16

        # model_name = "microsoft/resnet-50"
        # square = 16

        self.clip_model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        layer: transformers.models.clip.modeling_clip.CLIPEncoderLayer = self.clip_model.vision_model.encoder.layers[-1]
        target_layer: torch.nn.LayerNorm = layer.layer_norm1

        # Hook to capture feature maps during forward pass

        def save_features(module, input, output: torch.Tensor):
            self.feature_maps = output
            self.activations.append(reshape_transform(output, width=square, height=square).cpu().detach().numpy())

        # Hook to capture gradients during backward pass
        def save_gradients(module, grad_input, grad_output: torch.Tensor):
            self.gradients = grad_output
            self.gradients_array.append(reshape_transform(
                grad_output[0], width=square, height=square).cpu().detach().numpy())

        # Attach hooks to the target layer
        target_layer.register_forward_hook(save_features)
        target_layer.register_full_backward_hook(save_gradients)

    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)

        # Compute loss for specific class
        # This is where your understanding needs a slight modification
        # You create a loss based on the output, not directly backprop a label
        class_score = output[:, target_class]
        loss = class_score.sum()

        # Backward pass
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3))
        cam = (self.feature_maps * weights[:, :, None, None]).sum(dim=1)
        cam = torch.clamp(cam, min=0)
        cam = normalize_cam(cam)
        cam = resize_cam(cam, input_tensor.shape[-2:])

        # Overlay on original image
        overlaid_cam = overlay_cam(input_tensor, cam)
        return overlaid_cam

    def main(self, pil_img: Image):
        inputs: transformers.tokenization_utils_base.BatchEncoding = self.processor(
            images=pil_img,
            return_tensors="pt"  # Return PyTorch tensors
        )
        image_tensor: torch.Tensor = inputs['pixel_values']  # Shape: (1, 3, 224, 224)

        # output = self.clip_model.vision_model.forward(image_tensor)
        output = self.clip_model.get_image_features(image_tensor)
        # output = self.clip_model(**inputs).image_embeds
        self.clip_model.zero_grad()
        loss = self.text_label_loss(output)
        loss.backward()

        layer_activations = self.activations[0]
        layer_grads = self.gradients_array[0]

        weights = np.mean(layer_grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * layer_activations
        cam = weighted_activations.sum(axis=1)  # (1, 16, 16)
        cam = np.maximum(cam, 0)
        # cam = np.abs(cam)
        cam = scale_cam_image(cam, target_size=pil_img.size)
        # cam = np.mean(cam, axis=0)

        numpy_img = np.array(pil_img) / 255.0

        cam_image = show_cam_on_image(numpy_img, cam[0, :], image_weight=0.66)
        cv2.imwrite(f'nocab_cam.jpg', cam_image)
        pass

    def text_label_loss(self, output: torch.Tensor):
        inputs = self.tokenizer(text=["a fish", "a dog", "a cat", "bulldog", "grass"], padding=True, return_tensors="pt")
        # inputs = self.tokenizer(text=["an empty scene"], padding=True, return_tensors="pt")
        text_output = self.clip_model.get_text_features(**inputs)

        target_class_id = 1
        text_embedding = text_output[target_class_id]
        result = -torch.cosine_similarity(output.squeeze(0), text_embedding, dim=0)
        return result


def normalize_cam(cam):
    # Normalize CAM to [0,1] range
    pass


def resize_cam(cam, target_size):
    # Resize CAM to match input image
    pass


def overlay_cam(input_tensor, cam):
    # Blend CAM with original image
    pass


def self_similarity_cam(embedding_tensor):
    # Create a pseudo-loss based on self-similarity
    loss = -torch.cosine_similarity(
        embedding_tensor,
        embedding_tensor.detach(),
        dim=0
    )
    return loss


def clip_embedding_saliency(image_embedding):
    loss = torch.norm(image_embedding - image_embedding.detach())
    return loss


def main():
    pil_img: Image = Image.open("./img/dog.jpg")
    # start_time = time.time()
    gradcam = GradCAM()
    start_time = time.time()
    gradcam.main(pil_img)
    end_time = time.time()
    print(f"Generating frame took {end_time - start_time:.4f} seconds to execute")


if __name__ == "__main__":
    main()
