import os
import logging

APP_NAME = "kaggle-human-protein-atlas"
DATASET_PATH = os.getenv("DATASET_PATH", "./data/official/")
MODEL_PATH = os.getenv("MODEL_PATH", "./model/")
OUT_PATH = os.getenv("OUT_PATH", "./output/")

_log_format = "*** %(asctime)s - %(name)s - %(levelname)s ***\n%(message)s\n******\n"
logging.basicConfig(
                    format=_log_format,
                    level=logging.DEBUG,
                   )
formatter = logging.Formatter(_log_format)

WEIGHTS = [
    0.41468202883,
    0.04035787847,
    0.11653578784,
    0.05023815653,
    0.05979660144,
    0.08087667353,
    0.03244078269,
    0.09082131822,
    0.00170571575,
    0.00144824922,
    0.00090113285,
    0.03517636457,
    0.02214212152,
    0.01728244078,
    0.03430741503,
    0.00067584963,
    0.01705715756,
    0.00675849639,
    0.02902935118,
    0.04769567456,
    0.00553553038,
    0.12155638517,
    0.02581101956,
    0.09542353244,
    0.01036302780,
    0.26480432543,
    0.01055612770,
    0.00035401647,
]
