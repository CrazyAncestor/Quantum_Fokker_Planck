import os
import imageio
import argparse
# python create_animation.py <bin_folder>  --fps 10
# Set up argument parser
parser = argparse.ArgumentParser(description="Create a video from PNG images.")
parser.add_argument("bin_folder", help="Path to the folder containing PNG images.")
parser.add_argument("--fps", type=int, default=5, help="Frames per second for the video (default: 5).")

# Parse arguments
args = parser.parse_args()

# Define the path to your PNG files
bin_folder = args.bin_folder + str('Snapshots') # Folder containing PNG images
output_folder = args.bin_folder  # Folder to save the output video
fps = args.fps  # Frames per second

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all PNG files in the folder
png_files = [f for f in os.listdir(bin_folder) if f.endswith('.png')]

# Sort the files if needed (optional)
png_files.sort()

# Create a list to store the images
images = []

# Load each image and append it to the images list
for file in png_files:
    img_path = os.path.join(bin_folder, file)
    img = imageio.imread(img_path)
    images.append(img)

# Define the output file path
output_file = os.path.join(output_folder, 'animation.mp4')

# Save as MP4 (make sure ffmpeg is available)
imageio.mimsave(output_file, images, fps=fps)  # Adjust the FPS as needed

print(f"Video saved to: {output_file}")
