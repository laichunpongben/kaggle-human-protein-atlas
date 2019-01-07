import os
import logging

APP_NAME = "kaggle-human-protein-atlas"
DATASET_PATH = os.getenv("DATASET_PATH", "./data/hpa1024/")
MODEL_PATH = os.getenv("MODEL_PATH", "./model/")
PRED_PATH = os.getenv("PRED_PATH", "./pred/")
OUT_PATH = os.getenv("OUT_PATH", "./output/")
LOG_PATH = os.getenv("LOG_PATH", "./logs/")
PLOT_PATH = os.getenv("PLOT_PATH", "./plot/")

_log_format = "*** %(asctime)s - %(name)s - %(levelname)s ***\n%(message)s\n******\n"
logging.basicConfig(
                    format=_log_format,
                    level=logging.DEBUG,
                   )
formatter = logging.Formatter(_log_format)

# STATS = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])

STATS = {
    #  ds: mean, std
    # "official":        ([0.07986162506177984, 0.05217604947235713, 0.054227752481757215, 0.08201468927464939],
    #                     [0.1403192215484648, 0.1041239635111223, 0.1532386688507187, 0.14099509309392533]),
    # bs 32
    # "official":        ([0.08035214900985806, 0.052662718446443796, 0.054821162479790554, 0.08270705238092081],
    #                     [0.1441818831316468, 0.10766732147991964, 0.15509390386065397, 0.1447311870215956]),
    # bs 4
    "official":        ([0.08181731294659297, 0.05393953952823462, 0.05515322176105104, 0.08418251280411483],
                        [0.14912430271152502, 0.11185498801507318, 0.15687271707973055, 0.14903431519537524]),
    # bs 8
    # "official_hpav18": ([0.09268804824442385, 0.05610595307977252, 0.05822781634430247, 0.085418029487904],
    #                     [0.15618502296096703, 0.1101010284421363, 0.16142362852859507, 0.13949583832380028]),
    "hpav18":          ([0.11810116173772936, 0.06795416990341785, 0.06612933906943513, 0.08437984839655913],
                        [0.17499094128737935, 0.12197690061419421, 0.17568283970650891, 0.11266782268834614]),
    "official_hpav18": ([0.1066346176636245, 0.06331175038503216, 0.06268440253388274, 0.08384531792200918],
                        [0.167137549903456, 0.11800686037410352, 0.169741759561414, 0.12320841516204066])
}

# Reference of nuclei count and density is from the same image
BASE_NUCLEI_COUNT = 10  # arbitrary
BASE_NUCLEI_DENSITY = 0.5  # arbitrary

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
