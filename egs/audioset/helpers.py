import os
import sys
import csv
import argparse
import yaml

import numpy
import numpy as np
import torch
import torchaudio
from collections import OrderedDict
from copy import deepcopy
import global_vars as GLOBALS

import pandas as pd

torchaudio.set_audio_backend("sox_io")
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

from src.models import ASTModel

def initialize_hyper(path_to_config):
    '''
    Reads config.yaml to set hyperparameters
    '''
    with open(path_to_config, 'r') as stream:
        try:
            GLOBALS.CONFIG = yaml.safe_load(stream)
            return GLOBALS.CONFIG
        except yaml.YAMLError as exc:
            print(exc)
            return None

def set_up_parser():
    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audio_path ./0OxlgIitVig.wav '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')

    parser.add_argument("--model_path", type=str, required=True,
                        help="the trained model you want to test")
    parser.add_argument('--audio_path',
                        help='the audio you want to predict, sample rate 16k.',
                        type=str, required=True)
    return parser

def make_features(waveform, sr, mel_bins, target_length=1024):
    if len(waveform.shape)==1:
        waveform = waveform[None, :]
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    fbank = fbank.expand(1, target_length, mel_bins)
    return fbank

def load_label(label_csv='./data/class_labels_indices.csv'):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels

# def create_single_features(curr_waveform_1d, sr):
#

def forward_prop(curr_feats_data, sr, audio_model):
    curr_feats_data = curr_feats_data[None, :] if len(curr_feats_data.shape)!=3 else curr_feats_data
    with torch.no_grad():
        output = audio_model.forward(curr_feats_data)
        output = torch.sigmoid(output)
    return output

def create_single_output(wav_name, audio_model):
    waveform, sr = torchaudio.load(wav_name)
    curr_feats_data = make_features(waveform, sr, 128)
    curr_outputs = forward_prop(curr_feats_data, sr, audio_model)
    return curr_outputs

def create_all_outputs(wav_name,audio_model):
    waveform, sr = torchaudio.load(wav_name)
    duration = int(waveform.flatten().shape[0] / sr)
    increment = 5

    all_feats_data = None
    for start_point in np.arange(0, duration - increment+0.000001, increment):
        curr_waveform = waveform.flatten()[int(start_point * sr): int((start_point + increment) * sr)]
        curr_feats_data = make_features(curr_waveform, sr, 128)
        if type(all_feats_data) == torch.Tensor:
            all_feats_data = torch.cat([all_feats_data, curr_feats_data])
        else:
            all_feats_data = curr_feats_data
    all_outputs = forward_prop(all_feats_data, sr, audio_model)
    return all_outputs, increment, duration

def step_by_step(wav_name, audio_model, labels):
    all_outputs, increment, duration = create_all_outputs(wav_name, audio_model)
    output_dict = OrderedDict()
    for index, start_point in enumerate(np.arange(0, duration - increment, increment)):
        curr_result_output = all_outputs.data.cpu().numpy()[index]
        curr_sorted_indexes = np.argsort(curr_result_output)[::-1]
        stat_info = ["{} {}".format(np.array(labels)[curr_sorted_indexes[k]], round(float(curr_result_output[curr_sorted_indexes[k]]),3)) for k in range(3)]
        output_dict["{}-{} seconds".format(start_point,start_point+increment)] = stat_info
    return output_dict

def single_file_test(wav_name, audio_model, labels):
    waveform, sr = torchaudio.load(wav_name)
    duration = int(waveform.flatten().shape[0] / sr)
    curr_feats_data = make_features(waveform, sr, 128)


def set_up_model(checkpoint_path, input_tdim=1024):
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=True)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    return audio_model