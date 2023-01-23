from easydict import EasyDict
import yaml

with open("./config.yaml", "r") as stream:
    try:
        yaml_file = yaml.safe_load(stream)
    except yaml.YAMLError as exception:
        print(exception)

cfg = EasyDict()

cfg.SERVER = EasyDict()
cfg.SERVER.PORT = yaml_file["port"]
cfg.SERVER.VERSION = "1.1.0"
cfg.SERVER.TITLE = "MV Detection Server"
cfg.SERVER.DEBUG = True
cfg.SERVER.MODELS_PATH = "./pytorch_models"
cfg.SERVER.VIEWS_PER_PASS = yaml_file["views_per_pass"]
cfg.SERVER.NUMBER_OF_PASSES = yaml_file["number_of_passes"]
cfg.SERVER.NUMBER_OF_CLASSES = 2  # number of classes plus background
cfg.SERVER.CLASSES = ('__background__', 'explosive')
cfg.SERVER.DEVICE = 'cuda'
cfg.SERVER.MODEL_EXTENSION = "pth"
cfg.SERVER.IMAGES_FOR_INSPECTION_DIRECTORY = yaml_file["images_directory"]
cfg.SERVER.MODEL_THRESHOLD = yaml_file["model_threshold"]
