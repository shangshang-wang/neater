from PIL import Image
import os

# Folder with PNGs
image_folder = "./logs"
# Output GIF file
output_gif = "output.gif"

# Get all PNG files sorted by filename
images = [Image.open(os.path.join(image_folder, file))
          for file in sorted(os.listdir(image_folder))
          if file.endswith('.png')]

# Convert all to RGB or RGBA to ensure compatibility
images = [img.convert('RGBA') for img in images]

# Save as GIF
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=100,       # duration per frame in ms
    loop=0              # loop=0 for infinite loop
)

print(f"GIF saved to {output_gif}")