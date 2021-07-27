import glob
from PIL import Image
import sys
import os

folder = input("Enter folder name containing PNG's: ")
# folder = sys.argv[1]

fp_in = f'{folder}/*.png'
save_folder = os.path.join(f'{folder}', 'gif')
os.makedirs(save_folder, exist_ok=True, mode=0o755)
fp_out = f'{save_folder}/gif.gif'


# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500, loop=0, dpi=(300,300)) # duration in ms between each frame
print(f'GIF created at: {save_folder}')