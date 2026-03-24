# -*- mode: python ; coding: utf-8 -*-
# ==============================================================================
# TuNet PyInstaller Spec File
#
# Builds a --onedir distribution with CUDA/cuDNN support for GPU training.
# Entry point: tunet_main.py (dispatcher that routes to sub-scripts)
#
# Build with:  conda run -n tunet pyinstaller scripts/tunet.spec
# ==============================================================================

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- Project paths ---
PROJECT_DIR = os.path.abspath('.')

# --- Bundled Python scripts (executed via subprocess dispatch) ---
script_datas = [
    (os.path.join(PROJECT_DIR, 'ui_app.py'), '.'),
    (os.path.join(PROJECT_DIR, 'train.py'), '.'),
    (os.path.join(PROJECT_DIR, 'inference.py'), '.'),
    (os.path.join(PROJECT_DIR, 'inference_gui.py'), '.'),
    (os.path.join(PROJECT_DIR, 'inference_gui_multigpu.py'), '.'),
    (os.path.join(PROJECT_DIR, 'training_monitor.py'), '.'),
    (os.path.join(PROJECT_DIR, 'check_exr_dims.py'), '.'),
]

# --- Utility scripts (in utils/ subdirectory) ---
util_datas = [
    (os.path.join(PROJECT_DIR, 'utils', 'convert_flame.py'), 'utils'),
    (os.path.join(PROJECT_DIR, 'utils', 'convert_nuke.py'), 'utils'),
    (os.path.join(PROJECT_DIR, 'utils', 'pad_images.py'), 'utils'),
]

# --- Config / data files ---
config_datas = [
    (os.path.join(PROJECT_DIR, 'base', 'base.yaml'), 'base'),
    (os.path.join(PROJECT_DIR, 'config_templates'), 'config_templates'),
    (os.path.join(PROJECT_DIR, 'models', 'ACTS_model_config.yaml'), 'models'),
    (os.path.join(PROJECT_DIR, 'models', 'ACTS_model_config_matte.yaml'), 'models'),
    (os.path.join(PROJECT_DIR, 'models', 'ACTS_model_roto.yaml'), 'models'),
]

# --- Dataloader module (Python source needed for imports) ---
dataloader_datas = [
    (os.path.join(PROJECT_DIR, 'dataloader'), 'dataloader'),
]

# --- Collect package data ---
# lpips needs its pretrained weight files
lpips_datas = collect_data_files('lpips')

# albumentations may have version info
albumentations_datas = collect_data_files('albumentations')

# onnx data
onnx_datas = collect_data_files('onnx')

# Combine all data files
all_datas = (
    script_datas
    + util_datas
    + config_datas
    + dataloader_datas
    + lpips_datas
    + albumentations_datas
    + onnx_datas
)

# --- Hidden imports ---
# Modules that PyInstaller can't detect from static analysis
hidden_imports = [
    # Core project modules
    'ui_app',
    'train',
    'inference',
    'inference_gui',
    'inference_gui_multigpu',
    'training_monitor',
    'dataloader',
    'dataloader.data',
    'tunet_updater',

    # PyTorch ecosystem
    'torch',
    'torch.nn',
    'torch.optim',
    'torch.distributed',
    'torch.amp',
    'torch.nn.parallel',
    'torch.utils.data',
    'torch.utils.data.distributed',
    'torchvision',
    'torchvision.transforms',
    'torchvision.utils',
    'torchaudio',

    # CUDA backends
    'torch.backends.cudnn',
    'torch.cuda',
    'torch.cuda.amp',

    # Image processing
    'cv2',
    'PIL',
    'PIL.Image',
    'OpenEXR',
    'Imath',

    # Data / augmentation
    'albumentations',
    'albumentations.pytorch',
    'albumentations.pytorch.transforms',
    'albucore',

    # Loss functions
    'lpips',

    # Scientific
    'numpy',
    'scipy',
    'scipy.ndimage',

    # GUI frameworks
    'PySide6',
    'PySide6.QtWidgets',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.scrolledtext',

    # Visualization
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',

    # Config / serialization
    'yaml',
    'json',
    'argparse',

    # Export
    'onnx',
    'onnxruntime',

    # Other
    'pydantic',
    'coloredlogs',
    'tqdm',
    'tqdm.auto',
    'multiprocessing',
    'multiprocessing.spawn',
    'multiprocessing.popen_spawn_win32',

    # torch internals often missed
    'torch._C',
    'torch._utils',
    'torch.utils._pytree',
    'torch.distributed.constants',
    'torch.distributed.distributed_c10d',
    'torch.distributed.rendezvous',
    'torch.distributed.algorithms',
]

# Collect all submodules for tricky packages
hidden_imports += collect_submodules('torch.distributed')
hidden_imports += collect_submodules('PySide6')
hidden_imports += collect_submodules('albumentations')
hidden_imports += collect_submodules('lpips')
hidden_imports += collect_submodules('onnxruntime')

# Deduplicate
hidden_imports = list(set(hidden_imports))

# --- Binaries ---
# PyInstaller should auto-collect torch DLLs (including CUDA), but let's
# make sure by collecting torch binaries explicitly
torch_binaries = []
try:
    import torch as _torch
    torch_lib = os.path.join(os.path.dirname(_torch.__file__), 'lib')
    if os.path.isdir(torch_lib):
        for f in os.listdir(torch_lib):
            if f.endswith('.dll') or f.endswith('.so'):
                torch_binaries.append((os.path.join(torch_lib, f), 'torch/lib'))
except ImportError:
    pass

# --- Analysis ---
a = Analysis(
    [os.path.join(PROJECT_DIR, 'tunet_main.py')],
    pathex=[PROJECT_DIR],
    binaries=torch_binaries,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary large packages to save space
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
        'pip',
        'wheel',
        # pkg_resources pulls in jaraco, platformdirs, etc.
        # We don't need it at runtime
        'pkg_resources',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TuNet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Don't UPX compress - CUDA DLLs don't compress well and can break
    console=True,  # Keep console for training output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TuNet',
)
