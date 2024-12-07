from typing import List
import PIL.Image
import clip
from clip.model import CLIP, Bottleneck

import clip.model
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
from src.Maintained_Queue import Maintained_Queue


def reshape_transform(tensor: torch.Tensor, height=16, width=16):
    return tensor  # noop

    # Taken from example
    blab = tensor[:, 1:, :]  # [1, 2047, 7, 7])
    result = tensor[:, 1:, :].reshape(tensor.size(0), 7, 7, 2047)
    #   height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class Nocab_GradCAM:
    def __init__(self):
        # Register forward hook to capture feature maps
        self.forward_activations = []
        self.backward_gradients = []
        self.recent_image_embeddings = Maintained_Queue(100)

        self.feature_maps: torch.Tensor = None  # the outputs of the target layer in the forward pass
        self.gradients: torch.Tensor = None  # the gradients of the target layer in the backwards pass

        self.device = "cpu"
        self.cam_clip_threshold = 0.50

        clip.available_models()
        model: CLIP = None
        preprocess = None
        model, preprocess = clip.load("RN50", device=self.device)
        square = 16

        self.clip_model = model
        self.tokenizer = clip.tokenize
        self.processor = preprocess
        l: clip.model.Bottleneck = model.visual.layer4[2]
        target_layer = l.relu3

        # Hook to capture feature maps during forward pass
        def save_features(module, input, output: torch.Tensor):
            self.forward_activations.append(reshape_transform(
                output, width=square, height=square).cpu().detach().numpy())

        # Hook to capture gradients during backward pass
        def save_gradients(module, grad_input, grad_output: torch.Tensor):
            self.backward_gradients.append(
                reshape_transform(
                    grad_output[0].cpu().detach().numpy(),
                    width=square, height=square))

        # Attach hooks to the target layer
        target_layer.register_forward_hook(save_features)
        target_layer.inplace = False
        target_layer.register_full_backward_hook(save_gradients)

    def main(self, pil_img: Image):
        cam = self.get_cam(pil_img)
        numpy_img = np.array(pil_img) / 255.0
        numpy_img = numpy_img if numpy_img.shape[2] == 3 else numpy_img[:, :, 0:3]
        cam_image = show_cam_on_image(numpy_img, cam[0, :], image_weight=0.66)
        cv2.imwrite(f'nocab_cam.jpg', cam_image)

    def get_cam(self, pil_img: Image):
        self.forward_activations = []
        self.backward_gradients = []
        image_tensor: torch.Tensor = self.processor(pil_img).unsqueeze(0).to(self.device)
        # Shape: (1, 3, 224, 224)

        output = self.clip_model.visual.forward(image_tensor)
        self.recent_image_embeddings.push(output.cpu().detach().squeeze(0))
        self.clip_model.zero_grad()
        loss = self.text_label_loss(output)
        # loss = self.recent_average_loss(output)
        # loss = self.recent_average_delta_loss(output)
        # loss = self.constant_loss(output)
        loss.backward()

        layer_activations = self.forward_activations[0]
        layer_grads = self.backward_gradients[0]

        weights = np.mean(layer_grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * layer_activations
        cam = weighted_activations.sum(axis=1)  # (1, 16, 16)
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = np.where(cam >= self.cam_clip_threshold,
                       cam,
                       0)  # Select only the high gradient values
        # cam = np.abs(cam)
        cam = scale_cam_image(cam, target_size=pil_img.size)
        # cam = np.mean(cam, axis=0)

        return cam

    def text_label_loss(self, output: torch.Tensor):
        inputs = self.tokenizer(texts=["an empty scene", "background clutter", "something noteworthy",
                                "fish on the bottom of the ocean", "A snowy forest", "birds", "people"]).to(self.device)
        text_outputs = self.clip_model.encode_text(inputs)

        target_class_id = 3
        text_embedding = text_outputs[target_class_id]
        result = -torch.cosine_similarity(output.squeeze(0), text_embedding, dim=0)
        return result

    def recent_average_loss(self, output: torch.Tensor):
        result = -torch.cosine_similarity(output.squeeze(0),
                                          self.recent_image_embeddings.average_as_tensor(),
                                          dim=0)
        return result

    def recent_average_delta_loss(self, output: torch.Tensor):
        delta = self.recent_image_embeddings.average_as_tensor() - output
        result = delta.norm()
        return result


def main():
    pil_img: Image = Image.open("./fish2.png")
    # start_time = time.time()
    gradcam = Nocab_GradCAM()
    start_time = time.time()
    gradcam.main(pil_img)
    end_time = time.time()
    print(f"Generating frame took {end_time - start_time:.4f} seconds to execute")


if __name__ == "__main__":
    main()
