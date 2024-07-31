import os
from PIL import Image
import pyheif

def convert_heic_to_png(heic_file_path, output_directory):
    # Read the HEIC file
    heif_file = pyheif.read(heic_file_path)

    # Convert HEIF/HEIC to a PIL Image object
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the output file path
    base_filename = os.path.splitext(os.path.basename(heic_file_path))[0]
    output_file_path = os.path.join(output_directory, f"{base_filename}.png")

    # Save the image as PNG
    image.save(output_file_path, "PNG")
    print(f"Converted {heic_file_path} to {output_file_path}")

# Example usage
input_heic_path = "./IMG_5362.HEIC"
output_dir = "./IMG_5362.png"
convert_heic_to_png(input_heic_path, output_dir)

