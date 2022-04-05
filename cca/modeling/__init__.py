from .cca import CCA
ARCHITECTURES = {"CCA": CCA}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
