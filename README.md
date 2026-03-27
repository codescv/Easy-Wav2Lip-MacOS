# Easy-Wav2Lip-MacOS (uv tool version)

This is a refactored version of Easy-Wav2Lip, optimized for MacOS (Apple Silicon) and distributed as a `uv tool`.

## Features
- **Zero Configuration**: Models are automatically downloaded to `~/.cache/wav2lip` on first run.
- **Improved Compatibility**: Integrated monkeypatches for `basicsr` and `torchvision` compatibility issues.
- **One-Line Install**: Powered by `uv`.

## Installation

Install globally using `uv`:

```bash
uv tool install git+https://github.com/codescv/Easy-Wav2Lip-MacOS --python 3.10
```

## Usage

Once installed, you can run `wav2lip` from any directory:

### Basic Sync (Fast)
```bash
wav2lip --face "input_video.mp4" --audio "input_audio.wav" --outfile "output.mp4"
```

### High Quality (Improved Masking)
```bash
wav2lip --face "input_video.mp4" --audio "input_audio.wav" --outfile "output.mp4" --quality Improved
```

### Enhanced (Upscaling with GFPGAN)
```bash
wav2lip --face "input_video.mp4" --audio "input_audio.wav" --outfile "output.mp4" --quality Enhanced
```

## Parameters
- `--checkpoint_path`: Name of the Wav2Lip model (default: `Wav2Lip_GAN.pth`). Auto-downloads if missing.
- `--face`: Path to input video or image.
- `--audio`: Path to input audio or video file.
- `--outfile`: Path to save the result.
- `--quality`: `Fast`, `Improved`, or `Enhanced`.

---
*Based on the original [Easy-Wav2Lip](https://github.com/anothermartz/Easy-Wav2Lip) by anothermartz.*
