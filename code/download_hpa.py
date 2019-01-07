import os
import argparse
from multiprocessing import Pool
from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np
import requests
from PIL import Image

df = pd.read_csv('data/HPAv18RBGY_wodpl.csv')
colors = ['blue','red','green','yellow']
hpa_dir = Path('data/hpa1024/train')
hpa_dir.mkdir(parents=True, exist_ok=True)
exist_ids = sorted([f.split('.png')[0] for f in os.listdir(hpa_dir) if f.endswith('.png')])

parser = argparse.ArgumentParser()
parser.add_argument("-p","--process", help="Number of processes", type=int, default=0)
args = parser.parse_args()
p_count = args.process


def download(id_):
    for color in colors:
        if '_'.join([id_, color]) in exist_ids:
            # print('{}_{} Already exist'.format(id_, color))
            pass
        else:
            try:
                image_dir, _, image_id = id_.partition('_')
                url = f'http://v18.proteinatlas.org/images/{image_dir}/{image_id}_{color}.jpg'
                print(url)
                r = requests.get(url)
                img = np.array(Image.open(BytesIO(r.content)).resize((1024, 1024), Image.LANCZOS))
                img_gs = None
                if img.shape == (1024, 1024, 3):
                    if color == 'red':
                        img_gs = img[:, :, 0]
                    elif color == 'green':
                        img_gs = img[:, :, 1]
                    elif color == 'blue':
                        img_gs = img[:, :, 2]
                    else:
                        img_gs = (img[:, :, 0] + img[:, :, 1])/2
                else:
                    # 24089_si27_F4_11 blue is all black
                    img_gs = img

                img_gs = img_gs.astype(np.uint8)
                image = Image.fromarray(img_gs)
                image.save(hpa_dir / f'{id_}_{color}.png', format='png')
            except Exception as e:
                print(e)
                print(f'{id_}_{color} broke...')


if __name__ == '__main__':
    if p_count > 0:
        p = Pool(p_count)
    else:
        p = Pool()
    p.map(download, df.Id)
