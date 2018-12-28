import os
import pandas as pd
import numpy as np
import requests
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
from io import BytesIO

df = pd.read_csv('data/HPAv18RBGY_wodpl.csv')
colors = ['blue','red','green','yellow']
hpa_dir = Path('data/hpav18/train')
hpa_dir.mkdir(parents=True, exist_ok=True)
exist_ids = sorted([f.split('.png')[0] for f in os.listdir(hpa_dir) if f.endswith('.png')])

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
                img = np.array(Image.open(BytesIO(r.content)).resize((512, 512), Image.LANCZOS))
                img_gs = None
                if color == 'red':
                    img_gs = img[:, :, 0]
                if color == 'green':
                    img_gs = img[:, :, 1]
                if color == 'blue':
                    img_gs = img[:, :, 2]
                else:
                    img_gs = (img[:, :, 0] + img[:, :, 1])/2

                img_gs = img_gs.astype(np.uint8)
                image = Image.fromarray(img_gs)
                image.save(hpa_dir / f'{id_}_{color}.png', format='png')
            except Exception as e:
                print(e)
                print(f'{id_}_{color} broke...')

p = Pool()
p.map(download, df.Id)
