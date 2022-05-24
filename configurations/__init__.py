from easydict import EasyDict

from configurations import server_config, network_config

cfg = EasyDict()

# load server configuration
cfg.update(server_config.cfg)

# load neural_network configuration
cfg.update(network_config.cfg)
