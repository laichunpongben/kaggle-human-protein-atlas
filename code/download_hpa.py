import pandas as pd
import requests
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
from io import BytesIO

df = pd.read_csv('../HPAv18RBGY_wodpl.csv')

def download(id_):
    try:
        hpa_dir = Path('../HPAv18')
        hpa_dir.mkdir(parents=True, exist_ok=True)
        image_dir, _, image_id = id_.partition('_')
        url = f'http://v18.proteinatlas.org/images/{image_dir}/{image_id}_blue_red_green_yellow.jpg'
        r = requests.get(url)
        image = Image.open(BytesIO(r.content)).resize((512, 512), Image.LANCZOS)
        image.save(hpa_dir / f'{id_}.png', format='png')
    except:
        print(f'{id_} broke...')

p = Pool()
p.map(download, df.Id)
