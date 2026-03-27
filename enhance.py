import warnings
from gfpgan import GFPGANer
from easy_functions import CACHE_DIR, ensure_model
import os

warnings.filterwarnings("ignore")


def load_sr():
    model_path = ensure_model("GFPGANv1.4.pth")
    run_params = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return run_params


def upscale(image, properties):
    _, _, output = properties.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True
    )
    return output
