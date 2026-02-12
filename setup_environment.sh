#!/bin/bash
echo "============================================"
echo "TuNet Environment Setup (Linux/macOS)"
echo "============================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found!"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi

echo "Creating conda environment from environment.yml..."
echo ""
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Environment creation failed!"
    echo "If the environment already exists, you can update it with:"
    echo "  conda env update -f environment.yml --prune"
    echo ""
    exit 1
fi

echo ""
echo "============================================"
echo "Environment created successfully!"
echo "============================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate tunet"
echo ""
echo "Then you can:"
echo "  - Train models: python train.py --config your_config.yaml"
echo "  - Multi-GPU training: torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --config your_config.yaml"
echo "  - Run inference: python inference.py --checkpoint model.pth --input_dir images/ --output_dir output/"
echo "  - Launch UI: python ui_app.py"
echo "  - Multi-GPU inference: python inference_gui_multigpu.py"
echo "  - Monitor training: python training_monitor.py --output_dir your_output_folder"
echo ""
