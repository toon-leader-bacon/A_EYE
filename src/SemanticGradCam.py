from PIL import Image
from typing import List
import clip
import clip.model
import cv2
import numpy as np
import numpy.typing as npt

from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)

from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

from src.Maintained_Queue import Maintained_Queue
from src.Similarity import AveragePriorSimilarity, PriorSimilarity, TextSimilarity, dot_product_similarity, cosine_similarity, euclidean_distance


class SemanticGradCam:
    def __init__(self,
                 model: clip.model.CLIP,
                 preprocess,
                 target_layers,

                 cam_clip_threshold: float = 0.85,
                 text_filter: str = "",
                 cam_algorithm=GradCAM,
                 device="cpu"):
        self.model = model
        self.preprocess = preprocess
        self.target_layers = target_layers

        self.cam_clip_threshold = cam_clip_threshold
        self.text_similarity = TextSimilarity(text_filter, model, device=device, similarity_func=dot_product_similarity)

        self.cam_algorithm = cam_algorithm
        self.device = device
        self.prior_embeddings = Maintained_Queue(maxSize=100)

    def clip_cam(self, image: Image):
        input_tensor = build_tensor_from_image(image, self.preprocess, self.device)

        # current_embedding = self.model.visual.forward(input_tensor)
        # self.prior_embeddings.push(current_embedding.detach().numpy())
        # average = self.prior_embeddings.average_as_tensor()

        with self.cam_algorithm(model=self.model.visual,
                                target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[self.text_similarity],
                                aug_smooth=False,
                                eigen_smooth=False)

            grayscale_cam = grayscale_cam[0, :]  # Just the [width, height]
            grayscale_cam = np.where(grayscale_cam >= self.cam_clip_threshold,
                                     grayscale_cam,
                                     0)  # Select only the high gradient values
            width, height = image.size
            grayscale_cam = cv2.resize(grayscale_cam, (width, height))  # resize gradcam to match size of frame
            return grayscale_cam


class PreviousGradCam():
    def __init__(self,
                 model: clip.model.CLIP,
                 preprocess,
                 target_layers,

                 cam_clip_threshold: float = 0.85,
                 #  text_filter: str = "",
                 cam_algorithm=GradCAM,
                 device="cpu"):
        self.model = model
        self.preprocess = preprocess
        self.target_layers = target_layers

        self.cam_clip_threshold = cam_clip_threshold
        # self.text_targets = [TextSimilarity(text_filter, model, device=device, similarity_func=dot_product_similarity)]
        self.cam_algorithm = cam_algorithm
        self.device = device

        # self.prior_embedding = None
        self.prior_embeddings = Maintained_Queue(maxSize=10)

    def clip_cam(self, image: Image):
        input_tensor = build_tensor_from_image(image, self.preprocess, self.device)
        # image_embedding = self.model.encode_image(input_tensor)

        current_embedding = self.model.visual.forward(input_tensor)
        self.prior_embeddings.push(current_embedding.detach().numpy())
        # if self.prior_embedding is None:
        #     self.prior_embedding = current_embedding

        with self.cam_algorithm(model=self.model.visual,
                                target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[AveragePriorSimilarity(self.prior_embeddings.average_as_tensor())],
                                aug_smooth=False,
                                eigen_smooth=False)

            grayscale_cam = grayscale_cam[0, :]  # Just the [width, height]
            grayscale_cam = np.where(grayscale_cam >= self.cam_clip_threshold,
                                     grayscale_cam,
                                     0)  # Select only the high gradient values
            width, height = image.size
            grayscale_cam = cv2.resize(grayscale_cam, (width, height))  # resize gradcam to match size of frame

            # self.prior_embedding = current_embedding
            return grayscale_cam


def merge_grad_frame(cam, pil_img):
    numpy_img = np.array(pil_img) / 255.0
    numpy_img = numpy_img if numpy_img.shape[2] == 3 else numpy_img[:, :, 0:3]
    cam_img = show_cam_on_image(numpy_img, cam[0, :], image_weight=0.66)
    return cam_img


def image_process_fast(grayscale_cam: npt.NDArray, image: npt.NDArray):
    cam_image = show_cam_on_image(
        image / 255.0,  # Normalize frame to range [0, 1]
        grayscale_cam,
        use_rgb=False,
        image_weight=0.8)
    # result = cv2.cvtColor(cam_image, cv2.COLORMAP_JET)  # [width, height, color_channels] (integers)
    return cam_image


def build_tensor_from_image(image: Image, preprocess, device):
    # Tensor of size [1, color_channels, width, height]
    return preprocess(image).unsqueeze(0).to(device)
