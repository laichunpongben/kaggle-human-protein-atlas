import pandas as pd
import requests
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
from io import BytesIO

df = pd.read_csv('HPAv18RBGY_wodpl.csv')
colors = ['blue','red','green','yellow']

def download(id_):
    for color in colors:
        try:
            hpa_dir = Path('data/HPAv18')
            hpa_dir.mkdir(parents=True, exist_ok=True)
            image_dir, _, image_id = id_.partition('_')
            url = f'http://v18.proteinatlas.org/images/{image_dir}/{image_id}_{color}.jpg'
            print(url)
            r = requests.get(url)
            image = Image.open(BytesIO(r.content)).resize((512, 512), Image.LANCZOS)
            image.save(hpa_dir / f'{id_}_{color}.png', format='png')
        except:
            print(f'{id_}_{color} broke...')

p = Pool()
p.map(download, df.Id)
