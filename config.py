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
