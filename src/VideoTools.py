from PIL import Image
from typing import Union
import torch
import numpy as np
from numpy import typing as npt
from vidgear.gears import CamGear
from pytubefix import YouTube
import cv2


def download_video(link):
    yt = YouTube(url=link, use_oauth=True)
    print(yt.title)
    print(yt.thumbnail_url)
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4')\
        .order_by('resolution').desc().first()
    video_stream.download()


def extract_frame(video_path, output_path, frame_number):
    """
    Extract a single frame from a video file and save it as an image.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    if success:
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} extracted and saved to {output_path}")
    else:
        print(f"Failed to extract frame {frame_number}")
    cap.release()


RESOLUTION_360 = "360p"
RESOLUTION_480 = "480p"
RESOLUTION_720 = "720p"
RESOLUTION_1080 = "1080p"


class StreamManager():
    def __init__(self, stream_url, is_stream=True, resolution: str = "360p"):
        options = {"STREAM_RESOLUTION": resolution}
        self.stream = CamGear(source=stream_url,
                              stream_mode=is_stream,
                              logging=False,
                              **options).start()

    def __del__(self):
        self.close()

    def get_current_frame(self) -> Union[cv2.typing.MatLike, None]:
        frame = self.stream.read()

        return frame

    def close(self):
        self.stream.stop()


def no_op(gradient, frame):
    return frame


def no_gradient(frame):
    return frame


def watch_stream_text_similarity(url_link, is_stream=True, get_gradient=no_gradient, process_frame=no_op, frames_per_grad=60, resolution="720p"):
    """Example use of the StreamManager class
    """
    stream = StreamManager(url_link, is_stream=is_stream, resolution=resolution)
    cv2.startWindowThread()
    cv2.namedWindow("Output Frame")

    # infinite loop
    frame_count = 0
    while True:
        frame = stream.get_current_frame()
        frame_img: Image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if frame is None:
            # break infinite loop
            break

        if frame_count % frames_per_grad == 0:
            gradient: npt.NDArray = get_gradient(frame_img)
        frame = process_frame(gradient, frame)

        # do something with frame here
        cv2.imshow("Output Frame", frame)
        cv2.imwrite(f'./img/gif_save/{frame_count}.png', frame)

        key = cv2.pollKey()
        # check for 'q' key-press
        if key == ord("q") or frame_count == 1247:
            # if 'q' key-pressed break out
            break

        frame_count += 1

    cv2.destroyAllWindows()
    stream.close()


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=OIqUka8BOS8"
    watch_stream(url)
