# Simple Model Image Encoder for Model Images
"""
This module is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""

'''
A simple U-Net encoder for model images. It transforms human-readable model images into a latent space representation suitable for further processing by a computer, while preserving the model face as a key feature.

Explanation:

Initialization:

in_channels: Number of input channels in the model image.
out_channels: Number of output channels in the latent space representation.
num_filters: Number of filters in the convolutional layers.

Convolutional Layers:

nn.Conv2d: Applies 2D convolutional layers to extract features from the image.
nn.BatchNorm2d: Normalizes the output of each convolutional layer to improve training stability.
nn.ReLU: Applies the ReLU activation function for non-linearity.

Max Pooling:

nn.MaxPool2d: Downsamples the feature maps by taking the maximum value within a 2x2 window.
Forward Pass:

The input model image is passed through a sequence of convolutional layers, batch normalization, and ReLU activation.

Max pooling is applied to downsample the feature maps and reduce the spatial resolution.
The output is a latent space representation that captures the essential features of the model image, including the face.

Preserving the Model Face:

The U-Net architecture is specifically designed for image segmentation tasks, where preserving spatial information is crucial.

By using a series of convolutional layers with appropriate padding and kernel sizes, the encoder can capture the details of the model face while extracting higher-level features.

The use of max pooling helps to maintain the relative positions of features, ensuring that the face remains identifiable in the latent space representation.

Additional Considerations:

The number of filters and the depth of the encoder can be adjusted based on the complexity of the model images and the desired level of detail in the latent space representation.

Pre-trained weights from a model trained on a similar task could be used to initialize the encoder, potentially improving performance.

The latent space representation can be used as input to a decoder network for tasks such as image reconstruction or generation.

This simple U-Net encoder provides a basic understanding of how to encode model images while preserving the face as a key feature. More advanced models may include additional layers, techniques, and attention mechanisms for improved performance and robustness.

'''

import torch
import torch.nn as nn


#Explanation:

# 1. ResidualBlock Class:

# Defines a residual block with two convolutional layers, batch normalization, and ReLU activation.
# The residual connection adds the input to the output of the convolutional layers, allowing the network to learn identity mappings and improve gradient flow.

# 2. ModelImageEncoder Class:

# Initializes the encoder with the specified input and output channels, and number of filters.
# Adds two ResidualBlock instances after the initial convolutional layers.
# The rest of the encoder architecture remains the same as before.

# Respecting Kaiming He's Contribution:

# Kaiming He is a prominent researcher in deep learning, known for his contributions to residual networks (ResNets).

# Adding residual blocks to the U-Net encoder incorporates the principles of ResNet architecture, which has been shown to improve training stability and performance.

# By using residual blocks, we acknowledge and respect Kaiming He's contribution to the field of deep learning.

# Additional Considerations:

# The number of residual blocks and their configuration can be adjusted based on the specific requirements of your task and computational resources.

# Consider exploring other advanced network architectures and techniques to further enhance the performance of your U-Net encoder.

# Experiment with different hyperparameters and training strategies to optimize the model for your specific dataset and application.

# By incorporating residual blocks into the U-Net encoder, we can improve its training stability, performance, and pay homage to the contributions of Kaiming He in the field of deep learning.

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.relu2(x)
        return x

class ModelImageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters):
        super(ModelImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()

        # Add residual blocks
        self.res_block1 = ResidualBlock(num_filters, num_filters)
        self.res_block2 = ResidualBlock(num_filters, num_filters)

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Pass through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x

# Using UnetEncoder for Fashion Model Image with Face Focus
## Here's how to use the provided UnetEncoder for a fashion model image with a focus on the face:

### 1. Import Libraries and Load Image:

import torch
from PIL import Image
import torchvision.transforms as transforms

# Load the fashion model image
input_model_image_path = "/Users/yinuo/Projects/suanfamama-multimodal/src/input/wei.png"
image = Image.open(input_model_image_path)

### 2. Preprocess Image:
'''
Resize the image to a suitable size for the U-Net encoder.
Convert the image to a PyTorch tensor.
Normalize the pixel values to the range [0, 1].
'''

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image)

### 3. Create UnetEncoder Instance:
'''
Define the input and output channels based on your specific needs.
Choose an appropriate number of filters for the convolutional layers.
'''

# Create UnetEncoder instance
encoder = ModelImageEncoder(in_channels=3, out_channels=64, num_filters=32)

### 4. Encode the Image:
'''
Pass the preprocessed image tensor through the UnetEncoder.
'''
# Encode the image
encoded_image = encoder(image_tensor.unsqueeze(0))

### 5. Process the Encoded Representation:
'''
The encoded_image will be a tensor containing the latent space representation of the fashion model image.
You can further process this representation depending on your specific application.
'''
# Print the shape of the encoded image
# print(encoded_image.shape)

print("Summary")
print("Input:", input_model_image_path)
print("Computation:", "Suanfamama Model Image Encoder Module")
print("Output:", encoded_image.shape)
#print("Output:", encoded_image)

'''
Summary
Input: /Users/yinuo/Projects/suanfamama-multimodal/src/input/wei.png
Output: torch.Size([1, 32, 128, 192])
'''

'''
Additional Considerations:

If the fashion model image is not already centered on the face, we need consider using techniques like face detection or cropping to ensure the face is prominent in the input image.

The choice of input and output channels, number of filters, and other hyperparameters may need to be adjusted based on the specific characteristics of your fashion model images and the desired outcome.

Consider exploring advanced U-Net architectures or incorporating attention mechanisms to further enhance the model's ability to focus on the face region.

By following these steps and adapting them to your specific requirements, we can effectively use the UnetEncoder to extract a latent space representation of a fashion model image while preserving the face as a key feature. This representation can then be used for various downstream tasks such as image reconstruction, generation, or analysis.
'''