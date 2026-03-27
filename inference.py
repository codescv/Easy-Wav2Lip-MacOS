import sys
import os

# Monkeypatch torchvision before any other imports
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

import torch
# Global monkeypatch for torch.load to handle PyTorch 2.6+ weights_only=True default
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

print("\rloading torch       ", end="")
import torch

print("\rloading numpy       ", end="")
import numpy as np

print("\rloading Image       ", end="")
from PIL import Image

print("\rloading argparse    ", end="")
import argparse

print("\rloading configparser", end="")
import configparser

print("\rloading math        ", end="")
import math

print("\rloading os          ", end="")
import os

print("\rloading sys         ", end="")
import sys

# Monkeypatch basicsr to avoid compatibility issues with newer torchvision
try:
    # Try to import local degradations first
    import degradations
    import basicsr.data
    sys.modules['basicsr.data.degradations'] = degradations
    import basicsr.utils
    sys.modules['basicsr.utils.degradations'] = degradations
except ImportError:
    pass

print("\rloading subprocess  ", end="")
import subprocess

print("\rloading pickle      ", end="")
import pickle

print("\rloading cv2         ", end="")
import cv2

print("\rloading audio       ", end="")
import audio

print("\rloading RetinaFace ", end="")
from batch_face import RetinaFace

print("\rloading re          ", end="")
import re

print("\rloading partial     ", end="")
from functools import partial

print("\rloading tqdm        ", end="")
from tqdm import tqdm

print("\rloading warnings    ", end="")
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)
print("\rloading upscale     ", end="")
from enhance import upscale

print("\rloading load_sr     ", end="")
from enhance import load_sr

from easy_functions import load_model, g_colab, ensure_model, CACHE_DIR

print("\rimports loaded!     ")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_device():
    if torch.cuda.is_available():
        return 'cuda', 0
    if torch.backends.mps.is_available():
        return 'mps', -1
    return 'cpu', -1

device, gpu_id = get_device()

if device == 'cpu':
    print('Warning: No GPU detected so inference will be done on the CPU which is VERY SLOW!')

predictor = None
mouth_detector = None

# creating variables to prevent failing when a face isn't detected
kernel = last_mask = x = y = w = h = None

g_colab_val = g_colab()

args = None
preview_window = "Both"

all_mouth_landmarks = []

model = detector = detector_model = None

def do_load(checkpoint_path):
    global model, detector, detector_model
    model = load_model(checkpoint_path)
    
    mobilenet_path = ensure_model("mobilenet.pth")
    detector = RetinaFace(
        gpu_id=gpu_id, model_path=mobilenet_path, network="mobilenet"
    )
    detector_model = detector.model

def face_rect(images):
    face_batch_size = 8
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size : (i + 1) * face_batch_size]
        all_faces = detector(batch)  # return faces list of all images
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = (tuple(map(int, box)), landmarks)
            yield prev_ret

def create_mask_from_landmarks(img, original_img, landmarks, coords, debug_idx=None):
    """
    Create a mouth mask using 5-point landmarks from RetinaFace.
    Landmarks indices: 0,1: eyes; 2: nose; 3,4: mouth corners.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape
    y1, y2, x1, x2 = coords

    if landmarks is not None:
        # Convert global landmarks to local face-crop coordinates
        # landmarks are absolute [x, y], coords are [y1, y2, x1, x2]
        m_left_global = landmarks[3]
        m_right_global = landmarks[4]
        
        m_left = np.array([m_left_global[0] - x1, m_left_global[1] - y1])
        m_right = np.array([m_right_global[0] - x1, m_right_global[1] - y1])
        
        # Estimate mouth center and width in local space
        m_center = (m_left + m_right) / 2.0
        m_width = np.linalg.norm(m_left - m_right)
        
        # Define mouth area polygon
        p1 = m_center + np.array([-0.7 * m_width, -0.3 * m_width])
        p2 = m_center + np.array([0.7 * m_width, -0.3 * m_width])
        p3 = m_center + np.array([0.7 * m_width, 0.6 * m_width])
        p4 = m_center + np.array([-0.7 * m_width, 0.6 * m_width])
        
        mouth_points = np.array([p1, p2, p3, p4], dtype=np.int32)
        
        # Create binary mask
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        
        # Feathering
        blur_size = int(m_width * args.mask_feathering / 100.0)
        if blur_size % 2 == 0: blur_size += 1
        if blur_size > 0:
            mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        else:
            mask_blurred = mask
            
        mask_pil = Image.fromarray(mask_blurred)

        # Debug visualization
        if debug_idx is not None and debug_idx < 5:
            debug_img = original_img.copy()
            # Draw landmarks
            for pt in [m_left, m_right]:
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            # Draw mouth box
            cv2.polylines(debug_img, [mouth_points], True, (255, 0, 0), 2)
            os.makedirs("temp/debug_frames", exist_ok=True)
            cv2.imwrite(f"temp/debug_frames/frame_{debug_idx}_mask.jpg", debug_img)
            cv2.imwrite(f"temp/debug_frames/frame_{debug_idx}_raw_mask.jpg", mask)
    else:
        # Fallback
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        mask[h_img//2:, :] = 255
        mask_pil = Image.fromarray(mask)

    input1 = Image.fromarray(img_rgb)
    input2 = Image.fromarray(orig_rgb)
    input2.paste(input1, (0, 0), mask_pil)
    
    res = cv2.cvtColor(np.array(input2), cv2.COLOR_RGB2BGR)
    return res, mask_pil

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes
            
def face_detect(images, results_file="last_detected_face.pkl"):
    # If results file exists, load it and return
    if os.path.exists(results_file):
        print("Using face detection data from last input")
        with open(results_file, "rb") as f:
            return pickle.load(f)

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    tqdm_partial = partial(tqdm, position=0, leave=True)
    for image, (rect_data) in tqdm_partial(
        zip(images, face_rect(images)),
        total=len(images),
        desc="detecting face in every frame",
        ncols=100,
    ):
        if rect_data is None:
            cv2.imwrite(
                "temp/faulty_frame.jpg", image
            )  # check this frame where the face was not detected.
            raise ValueError(
                "Face not detected! Ensure the video contains a face in all the frames."
            )

        rect, landmarks = rect_data
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append({"box": [x1, y1, x2, y2], "landmarks": landmarks})


    boxes_list = [r["box"] for r in results]
    boxes = np.array(boxes_list)
    if str(args.nosmooth) == "False":
        boxes = get_smoothened_boxes(boxes, T=5)
    
    final_results = []
    for i, image in enumerate(images):
        x1, y1, x2, y2 = boxes[i]
        final_results.append([image[int(y1):int(y2), int(x1):int(x2)], (int(y1), int(y2), int(x1), int(x2)), results[i]["landmarks"]])

    # Save results to file
    with open(results_file, "wb") as f:
        pickle.dump(final_results, f)

    return final_results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch, landmarks_batch = [], [], [], [], []
    print("\r" + " " * 100, end="\r")
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2), None] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords, landmarks = face_det_results[idx]

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        landmarks_batch.append(landmarks)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch, landmarks_batch
            img_batch, mel_batch, frame_batch, coords_batch, landmarks_batch = [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield img_batch, mel_batch, frame_batch, coords_batch, landmarks_batch


mel_step_size = 16

def main():
    global args, preview_window
    
    # Clean up last face detection data to avoid conflicts between different videos
    if os.path.exists("last_detected_face.pkl"):
        os.remove("last_detected_face.pkl")
    
    parser = argparse.ArgumentParser(
        description="Inference code to lip-sync videos in the wild using Wav2Lip models"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Name of saved checkpoint to load weights from",
        default="Wav2Lip_GAN.pth",
    )

    parser.add_argument(
        "--segmentation_path",
        type=str,
        default=os.path.join(CACHE_DIR, "face_segmentation.pth"),
        help="Name of saved checkpoint of segmentation network",
        required=False,
    )

    parser.add_argument(
        "--face",
        type=str,
        help="Filepath of video/image that contains faces to use",
        required=True,
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Filepath of video/audio file to use as raw audio source",
        required=True,
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Video path to save result. See default for an e.g.",
        default="results/result_voice.mp4",
    )

    parser.add_argument(
        "--static",
        type=bool,
        help="If True, then use only first video frame for inference",
        default=False,
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Can be specified only if input is a static image (default: 25)",
        default=25.0,
        required=False,
    )

    parser.add_argument(
        "--pads",
        nargs="+",
        type=int,
        default=[0, 10, 0, 0],
        help="Padding (top, bottom, left, right). Please adjust to include chin at least",
    )

    parser.add_argument(
        "--wav2lip_batch_size", type=int, help="Batch size for Wav2Lip model(s)", default=1
    )

    parser.add_argument(
        "--out_height",
        default=480,
        type=int,
        help="Output video height. Best results are obtained at 480 or 720",
    )

    parser.add_argument(
        "--crop",
        nargs="+",
        type=int,
        default=[0, -1, 0, -1],
        help="Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. "
        "Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width",
    )

    parser.add_argument(
        "--box",
        nargs="+",
        type=int,
        default=[-1, -1, -1, -1],
        help="Specify a constant bounding box for the face. Use only as a last resort if the face is not detected."
        "Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).",
    )

    parser.add_argument(
        "--rotate",
        default=False,
        action="store_true",
        help="Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg."
        "Use if you get a flipped result, despite feeding a normal looking video",
    )

    parser.add_argument(
        "--nosmooth",
        type=str,
        default=False,
        help="Prevent smoothing face detections over a short temporal window",
    )

    parser.add_argument(
        "--no_seg",
        default=False,
        action="store_true",
        help="Prevent using face segmentation",
    )

    parser.add_argument(
        "--no_sr", default=False, action="store_true", help="Prevent using super resolution"
    )

    parser.add_argument(
        "--sr_model",
        type=str,
        default="gfpgan",
        help="Name of upscaler - gfpgan or RestoreFormer",
        required=False,
    )

    parser.add_argument(
        "--fullres",
        default=3,
        type=int,
        help="used only to determine if full res is used so that no resizing needs to be done if so",
    )

    parser.add_argument(
        "--debug_mask",
        type=str,
        default=False,
        help="Makes background grayscale to see the mask better",
    )

    parser.add_argument(
        "--preview_settings", type=str, default=False, help="Processes only one frame"
    )

    parser.add_argument(
        "--mouth_tracking",
        type=str,
        default=False,
        help="Tracks the mouth in every frame for the mask",
    )

    parser.add_argument(
        "--mask_dilation",
        default=150,
        type=float,
        help="size of mask around mouth",
        required=False,
    )

    parser.add_argument(
        "--mask_feathering",
        default=151,
        type=int,
        help="amount of feathering of mask around mouth",
        required=False,
    )

    parser.add_argument(
        "--quality",
        type=str,
        help="Choose between Fast, Improved and Enhanced",
        default="Fast",
    )

    args = parser.parse_args()

    # Ensure segmentation model exists if needed
    if not args.no_seg:
        ensure_model("face_segmentation.pth")

    # Ensure temp and results directories exist
    os.makedirs("temp", exist_ok=True)
    out_dir = os.path.dirname(args.outfile)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if not g_colab_val:
        # Load the config file
        config = configparser.ConfigParser()
        config_path = os.path.join(ROOT_DIR, 'config.ini')
        if os.path.exists(config_path):
            config.read(config_path)
            # Get the value of the "preview_window" variable
            if config.has_section('OPTIONS'):
                preview_window = config.get('OPTIONS', 'preview_window')

    do_load(args.checkpoint_path)

    args.img_size = 96

    if os.path.isfile(args.face) and args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    elif args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        if args.fullres != 1:
            print("Resizing video...")
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if args.fullres != 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(
                    frame, (int(args.out_height * aspect_ratio), args.out_height)
                )

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    if not args.audio.endswith(".wav"):
        print("Converting audio to .wav")
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                args.audio,
                "temp/temp.wav",
            ]
        )
        args.audio = "temp/temp.wav"

    print("analysing audio...")
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )

    mel_chunks = []

    mel_idx_multiplier = 80.0 / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    full_frames = full_frames[: len(mel_chunks)]
    if str(args.preview_settings) == "True":
        full_frames = [full_frames[0]]
        mel_chunks = [mel_chunks[0]]
    print(str(len(full_frames)) + " frames to process")
    batch_size = args.wav2lip_batch_size
    if str(args.preview_settings) == "True":
        gen = datagen(full_frames, mel_chunks)
    else:
        gen = datagen(full_frames.copy(), mel_chunks)

    run_params = None
    if not args.quality == "Fast":
        if not args.quality == "Improved":
            print("Loading", args.sr_model)
            run_params = load_sr()

    for i, (img_batch, mel_batch, frames, coords, landmarks_batch) in enumerate(
        tqdm(
            gen,
            total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
            desc="Processing Wav2Lip",
            ncols=100,
        )
    ):
        if i == 0:
            print("Starting...")
            frame_h, frame_w = full_frames[0].shape[:-1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c, l in zip(pred, frames, coords, landmarks_batch):
            y1, y2, x1, x2 = c

            if str(args.debug_mask) == "True":
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            cf = f[y1:y2, x1:x2]

            if args.quality == "Enhanced":
                p = upscale(p, run_params)

            if args.quality in ["Enhanced", "Improved"]:
                # Use RetinaFace 5-point landmarks for masking, with coordinate correction
                p, last_mask = create_mask_from_landmarks(p, cf, l, c, debug_idx=i)

            f[y1:y2, x1:x2] = p

            if not g_colab_val:
                #cv2.imshow("face preview", p)
                pass

            if str(args.preview_settings) == "True":
                cv2.imwrite("temp/preview.jpg", f)
            else:
                out.write(f)

    out.release()

    if str(args.preview_settings) == "False":
        print("converting to final video")

        subprocess.check_call([
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            "temp/result.mp4",
            "-i",
            args.audio,
            "-c:v",
            "libx264",
            args.outfile
        ])

if __name__ == "__main__":
    main()
