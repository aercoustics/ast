"""
Deploy Sagemaker Endpoint

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
3. Run `python deploy-sagemaker.py endpoint_name` where endpoint_name is the name of your new endpoint
    
"""
import sys
from sagemaker.pytorch import PyTorchModel
from sagemaker import Session
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig

def upload_model(model_path="model.tar.gz"):
    sess = Session()
    model_data = sess.upload_data(
            path=model_path, bucket=sess.default_bucket(), key_prefix="model/pytorch"
        )
    model = PyTorchModel(
        entry_point="code/inference.py",
        source_dir="code/",
        role='sagemaker-execution',
        model_data=model_data,
        framework_version="1.10",
        py_version="py38",
    )
    return model

def deploy_model(model, endpoint_name):
    serverless_inference_config = ServerlessInferenceConfig(memory_size_in_mb=6144, max_concurrency=20)
    predictor = model.deploy(endpoint_name=endpoint_name, serverless_inference_config=serverless_inference_config)
    return predictor
    

if __name__ == '__main__':
    args = sys.argv[1:]
    endpoint_name = sys.argv[-1] if len(args)==1 else None

    model = upload_model()
    deploy_model(model, endpoint_name)