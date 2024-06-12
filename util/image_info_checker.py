# Fashion Image Info checker
"""
This util is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""

from PIL import Image

def get_image_resolution(image_path):
  """
  Outputs the resolution of a given image.

  Args:
    image_path: The path to the image file.

  Returns:
    A tuple containing the width and height of the image.
  """
  image = Image.open(image_path)
  width, height = image.size
  return width, height

# Example usage:
image_path = "/Users/yinuo/Projects/suanfamama-multimodal/src/input/ToyFashionDataset/cloth/00000_00.jpg"  # Replace with the actual path to your image
resolution = get_image_resolution(image_path)
print(f"Image Resolution: {resolution}")