# TuNet Setup Guide

Quick setup instructions for getting TuNet running on Windows, Linux, and macOS.

## Prerequisites

- **Miniconda/Anaconda**: [Download here](https://docs.conda.io/en/latest/miniconda.html)
- **NVIDIA GPU** (recommended for training): Requires CUDA-compatible GPU
- **Git** (to clone the repository)

## Installation

### Option 1: Automated Setup (Recommended)

**Windows:**
Simply double-click `setup_environment.bat` and it will:
1. Create a conda environment named "tunet"
2. Install all required dependencies
3. Show you the activation command

```batch
setup_environment.bat
```

**Linux/macOS:**
Run the shell script:
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### Option 2: Manual Setup

```batch
# Create conda environment
conda env create -f environment.yml

# Activate the environment
conda activate tunet
```

### Option 3: Using pip only

If you prefer pip without conda:

```batch
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Verifying Installation

After activation, test your setup:

```batch
conda activate tunet
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Training UI
Launch the graphical interface for training:
```batch
python ui_app.py
```

### 2. Training Monitor
Monitor training loss in real-time:
```batch
python training_monitor.py --output_dir path/to/your/output
```

### 3. Multi-GPU Inference
Run inference across multiple GPUs:
```batch
python inference_gui_multigpu.py
```

### 4. Command Line Training
```bash
# Single GPU
python train.py --config your_config.yaml

# Multi-GPU (Linux only - uses torchrun)
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --config your_config.yaml
```

### 5. Command Line Inference
```batch
python inference.py --checkpoint model.pth --input_dir images/ --output_dir output/
```

## GPU Setup Notes

### For CUDA 12.x (Latest GPUs)
The environment.yml installs PyTorch with CUDA 12.x support by default.

### For CUDA 11.x (Older GPUs)
If you have an older GPU, install PyTorch with CUDA 11.8:
```batch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only
For CPU-only systems (not recommended for training):
```batch
pip install torch torchvision torchaudio
```

## EXR File Support

TuNet supports high dynamic range (HDR) EXR files for VFX workflows:

- **Automatic tone mapping**: EXR files are tone-mapped using Reinhard operator
- **Preserves HDR detail**: Brings HDR values into displayable range
- **All inference tools support EXR**: CLI, GUI, and Multi-GPU

The setup automatically installs OpenEXR support. If you encounter issues:
```batch
pip install OpenEXR==3.2.4
```

## Troubleshooting

### "conda: command not found"
- Make sure Miniconda/Anaconda is installed
- Restart your terminal after installation
- Check that conda is in your PATH

### "CUDA not available"
- Verify you have an NVIDIA GPU: `nvidia-smi`
- Update GPU drivers: [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
- Reinstall PyTorch with correct CUDA version

### Import errors
```batch
# Reinstall dependencies
conda activate tunet
pip install -r requirements.txt --force-reinstall
```

## Updating the Environment

If dependencies are updated, refresh your environment:

```batch
conda activate tunet
conda env update -f environment.yml --prune
```

Or with pip:
```batch
pip install -r requirements.txt --upgrade
```

## Support

For issues and questions:
- GitHub Issues: [https://github.com/emlcpfx/tunet/issues](https://github.com/emlcpfx/tunet/issues)
- Original Project: [https://github.com/tpc2233/tunet](https://github.com/tpc2233/tunet)
