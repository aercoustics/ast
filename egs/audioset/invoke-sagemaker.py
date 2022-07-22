"""
Invoke Sagemaker Endpoint

To run: run `python invoke-sagemaker.py` or `python invoke-sagemaker.py -save_res` (to save results in csv)
Runs on all files added to "egs/audioset/wav_files" folder

Currently working sagemaker endpoint is "endpoint-2022-06-30-6GB"
"""

import pyquist as pq
import pandas as pd
import boto3
import json
import os
import sys
from tqdm import tqdm
from helpers import *


def evaluate_results(res, save_res, filename):
    labels = load_label()
    sorted_indexes = np.argsort(res[0])[::-1]
    df = pd.DataFrame(columns=['class', 'confidence'])
    if save_res: 
        print('Saving results...')
    else:
        print('[*INFO] predice results:')
    for k in range(10):
        if save_res:
            df = pd.concat([df, pd.DataFrame(data=[[np.array(labels)[sorted_indexes[k]],res[0][sorted_indexes[k]]]], 
                                             columns=df.columns, index=[filename])])
        else:
            print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                    res[0][sorted_indexes[k]]))
    if save_res:
        pq.io.write_csv(filename.stem+'.csv', df)
    
def make_features(wav_name, mel_bins, target_length=1024):
    '''
    Similar to make_features in helpers.py with minor edits for functionality
    '''
    waveform, sr = torchaudio.load(wav_name)
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


if __name__ == '__main__':

    args = sys.argv[1:]
    save_res = True if len(args)==1 and args[0]=='-save_res' else False
    
    for filename in tqdm(list(os.listdir("wav_files"))):
        filepath = os.path.join("wav_files", filename)
        filename = pq.io.Path(filename)

        feats = make_features(filepath, mel_bins=128)           # shape(1024, 128)
        input_tdim = feats.shape[0]
        feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature

        body = json.dumps({"instances": feats.numpy().astype(float).tolist()})

        # TODO: Explore ways to send several file info at the same time rather than invoke endpoint for each
        client = boto3.client('sagemaker-runtime') 
        content_type = 'application/json'   

        # TODO: Parameterize name of working endpoint
        endpoint = 'endpoint-2022-06-30-6GB'
        response = client.invoke_endpoint(
            EndpointName=endpoint,
            Body=body,
            ContentType=content_type
        )
        predictions = response['Body']

        res = json.load(predictions)
        evaluate_results(res, save_res, filename)
        

