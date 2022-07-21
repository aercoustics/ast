import os
import torch
import json

# Network definition
from model_def import ASTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    ast_mdl = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
    model_path = os.path.join(model_dir, "model")
    print("In model_fn. Model path is -")
    print(model_path)
    with open(os.path.join(model_path , 'model.pth'), "rb") as f:
        print("Loading the ast model")
        checkpoint = torch.load(f, map_location=device)
        audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
        audio_model.load_state_dict(checkpoint)
    return audio_model


def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = json.loads(request_body)['instances']
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = data.expand(1, data.shape[0], 128)         # reshape the feature
    return data

def predict_fn(input_object, model):
    model.eval()                                      # set the eval model
    with torch.no_grad():
        output = model.forward(input_object)
        output = torch.sigmoid(output)
    return output

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)