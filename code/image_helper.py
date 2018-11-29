import os
import numpy as np
import PIL
from PIL import Image

def get_ids(path):
    return list(set(f.split('_')[0] for f in os.listdir(path) if f.endswith("png")))

def merge_rgb(id, input_path, output_path):
    colors = ['red', 'green', 'blue']
    convert_mode = 'L'

    rgb = [PIL.Image.open(os.path.join(input_path, "{}_{}.png".format(id, color))).convert(convert_mode) for color in colors]
    rgb = np.stack(rgb, axis=-1)  # channel last
    img = Image.fromarray(rgb)
    img.save(os.path.join(output_path, "{}.png".format(id)))


if __name__ == '__main__':
    input_path = "data/official/train"
    output_path = "data/rgb/train"
    ids = get_ids(input_path)
    print(len(ids))
    for id_ in ids:
        merge_rgb(id_, input_path, output_path)
