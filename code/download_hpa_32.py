import os
import pandas as pd
import requests
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
from io import BytesIO

df = pd.read_csv('data/HPAv18RBGY_wodpl.csv')
colors = ['blue','red','green','yellow']
hpa_dir = Path('data/hpav18')
hpa_dir.mkdir(parents=True, exist_ok=True)
exist_ids = sorted([f.split('.png')[0] for f in os.listdir(hpa_dir) if f.endswith('.png')])

def download(id_):
    for i,color in enumerate(colors):
        if '_'.join([id_, color]) in exist_ids:
            # print('{}_{} Already exist'.format(id_, color))
            pass
        else:
            try:
                image_dir, _, image_id = id_.partition('_')
                url = f'http://v18.proteinatlas.org/images/{image_dir}/{image_id}_{color}.jpg'
                if i%1000==0:
                    print(image_id)
                r = requests.get(url)
                image = Image.open(BytesIO(r.content)).resize((32, 32), Image.LANCZOS).convert('L')
                image.save(hpa_dir / f'{id_}_{color}.png', format='png')
            except:
                print(f'{id_}_{color} broke...')
     print('Done')
p = Pool()
p.map(download, df.Id)
