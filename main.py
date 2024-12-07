
from Nocab_GradCam_resnet import Nocab_GradCAM
from PIL import Image
from src.SemanticGradCam import PreviousGradCam, SemanticGradCam, image_process_fast, merge_grad_frame
from src.VideoTools import RESOLUTION_480, RESOLUTION_720, RESOLUTION_360, watch_stream_text_similarity
from torchsummary import summary
import clip
import clip.model
import torch
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)


# file_name = "frame.jpg"
# text_filter = "gun alien halo"
# text_filter = "health pack red cross medical"
# text_filter = "tree alpine forest"

device = "cuda" if torch.cuda.is_available() else "cpu"
cam_algorithm = GradCAMPlusPlus

# live birds
# url_link = "https://www.youtube.com/watch?v=OIqUka8BOS8"
# is_stream=True

# trail cam
# url_link = "https://www.youtube.com/watch?v=F0GOOP82094"
# url_link = "https://www.youtube.com/watch?v=oI8R4_UG3Fs"

# norway rail cam
# url_link = "https://www.youtube.com/watch?v=czoEAKX9aaM"

# fish
url_link = "https://www.youtube.com/watch?v=vO71aZ1xrFE"
# is_stream = False


# Halo
# url_link = "https://www.youtube.com/watch?v=W0v5vdEagzY&t=3s"
# Valkyria Chronicles
# url_link = "https://www.youtube.com/watch?v=LG34bWg8VOs"
# url_link = "https://www.youtube.com/watch?v=4o3ww8wxTBoq"

# tokyo
# url_link = "https://www.youtube.com/watch?v=0nTO4zSEpOs"

def main():
    cam_factory = Nocab_GradCAM()

    watch_stream_text_similarity(url_link=url_link,
                                 is_stream=True,
                                 get_gradient=cam_factory.get_cam,
                                 process_frame=merge_grad_frame,
                                 frames_per_grad=1,
                                 resolution=RESOLUTION_360)


# def main_old():
#     clip.available_models()
#     model, preprocess = clip.load("RN50", device=device)
#     target_layers = [model.visual.layer4]
#     # layer: clip.model.Bottleneck = model.visual.layer4[2]
#     # target_layers = [layer.relu3]

#     sgc = SemanticGradCam(
#         model=model,
#         preprocess=preprocess,
#         target_layers=target_layers,

#         cam_clip_threshold=0.75,
#         text_filter=text_filter,
#         cam_algorithm=HiResCAM,
#         device=device
#     )

#     watch_stream_text_similarity(url_link=url_link,
#                                  is_stream=True,
#                                  get_gradient=sgc.clip_cam,
#                                  process_frame=image_process_fast,
#                                  frames_per_grad=1,
#                                  resolution=RESOLUTION_480)


if __name__ == "__main__":
    main()
