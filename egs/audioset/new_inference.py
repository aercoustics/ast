import os
import sys
import csv
import argparse

import numpy as np
import torch
import torchaudio
from collections import OrderedDict

import pandas as pd

torchaudio.set_audio_backend("sox_io")
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'

from src.models import ASTModel
from helpers import *
import global_vars as GLOBALS



if __name__=="__main__":
    initialize_hyper("config.yaml")
    parser = set_up_parser()
    args = parser.parse_args()
    labels = load_label()
    audio_model = set_up_model(args.model_path)
    output_dict = step_by_step(args.audio_path, audio_model, labels)
    final_dataframe = pd.DataFrame.from_dict(output_dict)
    full_filename = args.audio_path.split("/")[0] + os.sep + "excel_analysis" + os.sep + args.audio_path.split("/")[1]
    filename = "{}_analysis.xlsx".format(full_filename)
    final_dataframe.to_excel(filename)
    print(final_dataframe.head())
    print(filename)