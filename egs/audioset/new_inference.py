import os
import sys
import csv
import argparse

import numpy as np
import torch
import torchaudio
from collections import OrderedDict

import pandas as pd

torchaudio.set_audio_backend("sox")
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'

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

def step_by_step(wav_name, audio_model, input_tdim=1024):
    waveform, sr = torchaudio.load(wav_name)
    duration = int(waveform.flatten().shape[0]/sr)
    increment = 5
    audio_model.eval()
    output_dict = OrderedDict()
    for start_point in np.arange(0, duration - increment, increment):
        curr_waveform = waveform.flatten()[int(start_point * sr) : int((start_point + increment) * sr)]
        curr_waveform = curr_waveform[None, :]
        curr_feats = make_features(curr_waveform, sr, mel_bins=128)
        curr_input_tdim = curr_feats.shape[0]
        curr_feats_data = curr_feats.expand(1, input_tdim, 128)
        with torch.no_grad():
            output = audio_model.forward(curr_feats_data)
            output = torch.sigmoid(output)
        curr_result_output = output.data.cpu().numpy()[0]
        curr_sorted_indexes = np.argsort(curr_result_output)[::-1]
        stat_info = ["{} {}".format(np.array(labels)[curr_sorted_indexes[k]], round(float(curr_result_output[curr_sorted_indexes[k]]),3)) for k in range(3)]
        output_dict["{}-{} seconds".format(start_point,start_point+increment)] = stat_info
    return output_dict


def set_up_model(args, input_tdim=1024):
    checkpoint_path = args.model_path
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=True)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    return audio_model


if __name__=="__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    labels = load_label()
    audio_model = set_up_model(args)
    output_dict = step_by_step(args.audio_path, audio_model)
    final_dataframe = pd.DataFrame.from_dict(output_dict)
    full_filename = args.audio_path.split("/")[0] + os.sep + "excel_analysis" + os.sep + args.audio_path.split("/")[1]
    filename = "{}_analysis.xlsx".format(full_filename)
    final_dataframe.to_excel(filename)
    print(final_dataframe.head())
    print(filename)