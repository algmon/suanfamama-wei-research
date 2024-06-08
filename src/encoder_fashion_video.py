# Simple Video Encoder especially for fashion videos
'''
# TODO: Fix this
Traceback (most recent call last):
  File "/Users/yinuo/Projects/suanfamama-multimodal/src/encoder_fashion_video.py", line 102, in <module>
    # Encode the video frames
                     ^^^^^^^^^
  File "/usr/local/Caskroom/miniforge/base/envs/algmon/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/Caskroom/miniforge/base/envs/algmon/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yinuo/Projects/suanfamama-multimodal/src/encoder_fashion_video.py", line 81, in forward
    # Convert the list of frames to a tensor
                          ^^^^^^^^^^^^^^^^^^^
TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Tensor
'''

'''
Explanation:

1. Import necessary libraries: 
* Import torch for tensor operations, torch.nn for neural network modules, torch.nn.functional for activation functions, and AutoTokenizer, AutoModelForSeq2SeqLM from the transformers library.

2. Function to load video frames:
* Imports the cv2 library for video processing.
* Loads the video using cv2.VideoCapture.
* Reads frames from the video in a loop.
* Appends each frame to a list.
* Releases the video capture object.
* Returns the list of frames.

3. VideoEncoder class:
* init: Initializes the model using the specified model_name_or_path.
* forward: Takes video_frames as input. Extracts features from the video frames using the pre-trained model. Returns the latent space representation of the video.

4. Example usage:
* Creates an instance of VideoEncoder with the desired pre-trained model.
* Loads the reference video frames using a function load_video_frames.
* Encodes the video frames using the VideoEncoder instance.
* Prints the latent space representation of the video.

Additional Notes:

* The model_name_or_path argument for the VideoEncoder should be a pre-trained model that is suitable for video feature extraction. Some examples include:
facebook/video-vit-base-patch32
facebook/video-vit-large-patch32
google/vit-base-patch32-384
google/vit-large-patch32-384

* The load_video_frames function is not provided here, as it depends on the specific video processing library you are using.

* The latent space representation of the video can be used for various downstream tasks, such as fashion item recognition, style transfer, and video captioning.

* Consider exploring the documentation and examples provided by the transformers library and the specific models you are using for more advanced usage and customization.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to load video frames
def load_video_frames(video_path):
    # Import necessary libraries
    import cv2

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Initialize an empty list to store the frames
    frames = []

    # Read frames from the video
    while True:
        ret, frame = cap.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Append the frame to the list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    # Return the list of frames
    return frames

class VideoEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super(VideoEncoder, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def forward(self, video_frames):
        # Convert the list of frames to a tensor
        video_frames_tensor = torch.stack(video_frames)

        # Extract features from the video frames using a pre-trained model
        video_features = self.model(video_frames_tensor)

        # Return the latent space representation of the video
        return video_features

# Example usage
video_encoder = VideoEncoder("t5-small")

# Load the reference video
video_frames = load_video_frames("/Users/yinuo/Projects/suanfamama-multimodal/src/input/mark.festival.2024.mp4")

# Convert the list of NumPy arrays to a list of tensors
video_frames_tensor = [torch.from_numpy(frame) for frame in video_frames]

# Stack the tensors into a single tensor
video_frames_tensor = torch.stack(video_frames_tensor)

# Encode the video frames
video_features = video_encoder(video_frames_tensor)

# Print the latent space representation of the video
print(video_features)