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

siren_file_set = set(list([
    "2021-06-24T23_41_47.294160+0000_sEVT_FW-D-M46B-M9B.wav",
    "2021-12-07T14_48_19.340268+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-07T14_45_56.340268+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-06T14_37_24.297468+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-06T05_49_29.187116+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-06T05_49_24.319929+0000_sEVT_SSE-LS1-M2.wav",
    "2021-12-06T05_01_34.187116+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-06T05_01_35.319929+0000_sEVT_SSE-LS1-M2.wav",
    "2021-12-06T04_39_31.187116+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-05T12_43_41.187116+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-05T05_32_29.259206+0000_sEVT_SSE-LS1-M2.wav",
    "2021-12-05T05_32_27.327031+0000_sEVT_SSE-LS1-M3.wav",
    "2021-12-05T00_42_16.259206+0000_sEVT_SSE-LS1-M2.wav",
    "2021-12-05T00_42_13.327031+0000_sEVT_SSE-LS1-M3.wav",
    "2021-09-27T03_30_44.230684+0000_sEVT_SSE-LS1-M3.wav",
    "2021-06-26T03_23_07.247935+0000_sEVT_FW-D-M46B-M9B.wav",
    "2021-06-25T23_07_46.311179+0000_sEVT_FW-D-M51.wav",
    "2021-06-25T23_07_02.145290+0000_sEVT_FW-D-M49-M25.wav",
    "2021-06-24T23_18_27.294160+0000_sEVT_FW-D-M46B-M9B.wav",
    "2021-06-24T23_26_39.380272+0000_sEVT_FW-C-M30.wav",
    "2021-06-24T23_27_18.312472+0000_sEVT_FW-C-M35.wav",
    "2021-06-24T23_32_54.278697+0000_sEVT_FW-A-M27-M6.wav",
    "2021-06-04T20_12_42.393397+0000_sEVT_FW-C-M31.wav",
    "2021-06-05T02_51_09.323145+0000_sEVT_FW-A-M27-M6.wav",
    "2021-06-18T23_26_38.401284+0000_sEVT_FW-D-M47B.wav",
    "2021-06-24T23_38_59.294160+0000_sEVT_FW-D-M46B-M9B.wav",
]))

initialize_hyper("config.yaml")

if __name__=="__main__":
    calculate_accuracy(desired_indices=GLOBALS.CONFIG["siren_indices"])
