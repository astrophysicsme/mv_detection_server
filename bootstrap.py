from os import path
import torch

from configurations import cfg
from lib.model.resnet import Resnet


def load_model(model_identifier: str, additional_parameters: dict = None):
    model_extension = cfg.SERVER.MODEL_EXTENSION
    model_name, model_type = model_identifier.split("_")
    model_file_name = f"{model_name}_{model_type}.{model_extension}"

    model_path = path.join(cfg.SERVER.MODELS_PATH, model_file_name)

    num_layers = model_name[6:]
    num_classes = cfg.SERVER.NUMBER_OF_CLASSES
    device = cfg.SERVER.DEVICE
    faster_rcnn = Resnet(num_layers, num_classes, class_agnostic=False)

    faster_rcnn.init()
    faster_rcnn.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    faster_rcnn.load_state_dict(checkpoint["model"])

    faster_rcnn.eval()

    return {
        "name": model_identifier,
        "model": faster_rcnn,
        "additional_parameters": additional_parameters
    }


def bootstrap_app():
    params = {
        "color_mode": "RGB",
        "image_range": 1,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    models = [load_model("resnet18_original", params), load_model("resnet34_original", params),
              load_model("resnet50_original"), load_model("resnet101_original")]

    return models


loaded_models = bootstrap_app()
