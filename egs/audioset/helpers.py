import os
import sys
import csv
import argparse
import yaml
import copy

import numpy
import numpy as np
import torch
import torchaudio
from collections import OrderedDict
from copy import deepcopy
import global_vars as GLOBALS
from sklearn.metrics import classification_report

import pandas as pd

torchaudio.set_audio_backend("sox_io")
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

from src.models import ASTModel


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

def conclude(wav_name, audio_model, desired_class="siren",primary_class_threshold = 3, secondary_class_threshold = 3, desired_indices = []):
    all_outputs, increment, duration = create_all_outputs(wav_name, audio_model)

    class_counter = 0
    for index, start_point in enumerate(np.arange(0, duration - increment+0.0000001, increment)):
        curr_result_output = all_outputs.data.cpu().numpy()[index]
        curr_sorted_indexes = np.argsort(curr_result_output)[::-1][:primary_class_threshold]
        for option in desired_indices:
            if option in curr_sorted_indexes:
                class_counter+=1
                break
    print("Evaluation Complete")
    return class_counter>=secondary_class_threshold


def calculate_accuracy(model_path="audioset_10_10_0.4593.pth", desired_indices = [], primary_class_threshold = 3, secondary_class_threshold = 3, precision_recall_setup=False, use_model=None):
    labels = load_label()
    if use_model is None:
        audio_model = set_up_model(model_path)
    else:
        audio_model = copy.deepcopy(use_model)
    audio_model.eval()
    correct, all = 0, 0
    actual_classifications, ground_classifications = [],[]
    for filename in list(os.listdir("wav_files")):
        if filename.endswith(".wav"):
            siren_or_not = conclude(os.path.join("wav_files", filename), audio_model, desired_indices=desired_indices, primary_class_threshold=primary_class_threshold, secondary_class_threshold=secondary_class_threshold)
            correct_class = "siren" if filename.split("/")[-1] in siren_file_set else "other"
            actual_class = "siren" if siren_or_not is True else "other"
            if actual_class == correct_class:
                correct += 1
            all += 1
            actual_classifications+=[actual_class]
            ground_classifications+=[correct_class]
            print("{}, Correct Class: {}, Actual Class: {}".format(100 * correct / all, correct_class, actual_class))
    all_stats = classification_report(ground_classifications,actual_classifications,target_names=GLOBALS.CONFIG["current_classes"],output_dict=True)
    if precision_recall_setup is True:
        return (100 * correct / all), all_stats
    else:
        return (100 * correct / all)