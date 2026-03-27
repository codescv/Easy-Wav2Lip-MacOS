import torch
import subprocess
import json
import os
import gdown
import pickle
import re
import sys
from models import Wav2Lip
from base64 import b64encode
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
from IPython.display import HTML, display

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model URLs - Using the most stable public sources
MODEL_URLS = {
    "Wav2Lip.pth": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip.pth",
    "Wav2Lip_GAN.pth": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth",
    "GFPGANv1.4.pth": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/GFPGANv1.4.pth",
    "face_segmentation.pth": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/face_segmentation.pth",
    "mobilenet.pth": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
}

# Local Cache Directory for models
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "wav2lip")

def ensure_model(filename):
    """Ensure the model file exists and is not an LFS pointer."""
    target_path = os.path.join(CACHE_DIR, filename)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Check if exists and is larger than 1MB (avoid LFS pointers)
    if not os.path.exists(target_path) or os.path.getsize(target_path) < 1024 * 1024:
        url = MODEL_URLS.get(filename)
        if not url:
            raise ValueError(f"Unknown model filename: {filename}")
            
        print(f"Model {filename} missing or invalid. Downloading from {url}...")
        try:
            # Using curl for better handling of redirects and large files
            subprocess.check_call(["curl", "-L", url, "-o", target_path])
        except Exception as e:
            print(f"Download failed: {e}")
            raise
                
    return target_path

def get_video_details(filename):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        filename,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = json.loads(result.stdout)

    # Get video stream
    video_stream = next(
        stream for stream in info["streams"] if stream["codec_type"] == "video"
    )

    # Get resolution
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    resolution = width * height

    # Get fps
    fps = eval(video_stream["avg_frame_rate"])

    # Get length
    length = float(info["format"]["duration"])

    return width, height, fps, length


def show_video(file_path):
    """Function to display video in Colab"""
    mp4 = open(file_path, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    width, _, _, _ = get_video_details(file_path)
    display(
        HTML(
            """
  <video controls width=%d>
      <source src="%s" type="video/mp4">
  </video>
  """
            % (min(width, 1280), data_url)
        )
    )


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _load(checkpoint_path):
    # Fix for PyTorch 2.6 default weights_only=True
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device(device), weights_only=False
        )
    except TypeError: # Older torch versions
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device(device)
        )
    return checkpoint


def load_model(path_or_name):
    # If it's just a filename, look in CACHE_DIR
    if not os.path.isabs(path_or_name) and not os.path.exists(path_or_name):
        filename = os.path.basename(path_or_name)
        path = ensure_model(filename)
    else:
        path = path_or_name

    folder, filename_with_extension = os.path.split(path)
    filename, file_type = os.path.splitext(filename_with_extension)
    results_file = os.path.join(folder, filename + ".pk1")
    
    if os.path.exists(results_file) and os.path.getsize(results_file) > 1024:
        with open(results_file, "rb") as f:
            return pickle.load(f)
            
    model = Wav2Lip()
    print("Loading {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    # Save results to file
    with open(results_file, "wb") as f:
        pickle.dump(model.eval(), f)
    return model.eval()


def get_input_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


def is_url(string):
    url_regex = re.compile(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$")
    return bool(url_regex.match(string))


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    if model_dir is None:
        model_dir = CACHE_DIR

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    
    # Check if exists and is larger than 1MB
    if not os.path.exists(cached_file) or os.path.getsize(cached_file) < 1024 * 1024:
        print(f'Downloading: "{url}" to {cached_file}\n')
        subprocess.check_call(["curl", "-L", url, "-o", cached_file])
    return cached_file


def g_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False
