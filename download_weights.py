import gdown
import wget
import zipfile
from pathlib import Path


Path("./resource/ckpt/acod").mkdir(parents=True, exist_ok=True)

effdet_urls = [
    'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth',
    'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth',
    'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth',
    'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth',
]

for url in effdet_urls:
    wget.download(url, out="./resource/ckpt")

gdown.download('https://drive.google.com/u/0/uc?export=download&confirm=XsPB&id=1K7MNVuW99uDMHciewVS71hks_YdU9_2A',
               'icpr2020.zip')
gdown.download('https://drive.google.com/u/0/uc?export=download&confirm=XsPB&id=1qdClbGL5KwgEc4oG023DJckQiidNHDF2',
               'acod_weights.zip')


with zipfile.ZipFile('icpr2020.zip', 'r') as zip_ref:
    files = [f for f in zip_ref.namelist() if f.startswith('resource/ckpt/ghnd/')]
    for f in files:
        zip_ref.extract(f, '.')

with zipfile.ZipFile('acod_weights.zip', 'r') as zip_ref:
    zip_ref.extractall('./resource/ckpt/acod')