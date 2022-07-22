# Aercoustics Implementation

Two files of note:
1) invoke-sagemaker.py
2) deploy-sagemaker.py

## invoke-sagemaker.py
To run: run `python invoke-sagemaker.py` (to print out results on console) or `python invoke-sagemaker.py -save_res` (to save results in csv in this directory) in this directory.

Runs on all files added to "egs/audioset/wav_files" folder, which users will need to create locally (this directory is added to the .gitignore so will not be a part of the git history)

Currently working sagemaker endpoint is "endpoint-2022-06-30-6GB" and this is hardcoded into the script. 


## deploy-sagemaker.py
Deploys sagemaker serverless endpoint. Review hardcoded parameters before running

To run: 
1. See `egs/audioset/model` folder. Add the model.pth file of your choice. Format should be as follows:
        model/
            |- model.pth
            |- code/
                |- inference.py
                |- model_def.py
                |- requirements.txt
2. Create a model.tar.gz file from the above directory
3. Run `python deploy-sagemaker.py endpoint_name` where endpoint_name is the name of your new endpoint. This is the name you will update to on the `invoke-sagemaker.py` script (currently set to "endpoint-2022-06-30-6GB")