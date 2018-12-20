import os
import logging

APP_NAME = "kaggle-human-protein-atlas"
DATASET_PATH = os.getenv("DATASET_PATH", "./data/official/")
MODEL_PATH = os.getenv("MODEL_PATH", "./model/")
OUT_PATH = os.getenv("OUT_PATH", "./output/")
LOG_PATH = os.getenv("LOG_PATH", "./logs/")

_log_format = "*** %(asctime)s - %(name)s - %(levelname)s ***\n%(message)s\n******\n"
logging.basicConfig(
                    format=_log_format,
                    level=logging.DEBUG,
                   )
formatter = logging.Formatter(_log_format)

# STATS = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
STATS = {
    #  ds: mean, std
    "official":        ([0.07986162506177984, 0.05217604947235713, 0.054227752481757215, 0.08201468927464939],
                        [0.1403192215484648, 0.1041239635111223, 0.1532386688507187, 0.14099509309392533]),
    "hpav18":          ([0.036928985010341434, 0.04130028252512823, 0.0075938375457115116, 0.0937920384196862],
                        [0.05419148261744557, 0.07329545732546368, 0.020430581830732493, 0.1444940434697745]),
    "official_hpav18": ([0.04952424341381588, 0.044563889728912606, 0.021262363915341524, 0.09030440845350635],
                        [0.09085177593717349, 0.08380707909661085, 0.08733944332031862, 0.1436538775485339])
}

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

THRESHOLDS = [
    0.0104,
    0.191,
    0.0091,
    0.0015,
    0.545,
    0.12,
    0.088,
    0.09,
    0.12,
    0.16,
    0.07,
    0.7,
    0.3,
    0.09,
    0.148,
    0.00025,
    0.102,
    0.269,
    0.026,
    0.074,
    0.0107,
    0.15,
    0.08,
    0.019,
    0.0031,
    0.051,
    0.056,
    0.015
]
