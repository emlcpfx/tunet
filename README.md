<div align="center">

# TUNET
A direct, pixel-level mapping from src to dst images via encoder-decoder.   
Supports, training, inference or export to Compositor tools such Foundry Nuke or Autodesk Flame native inference.


</div>


## 🎥 Watch In Action:
Flame:   
[![Flame video](https://img.youtube.com/vi/6-OFAJtfliM/hqdefault.jpg)](https://youtu.be/6-OFAJtfliM)

Nuke: (video soon)

## For Windows:   
✅ Make sure Miniconda or Anaconda is installed:
###[[Install Video](https://youtu.be/QaAca_LiwKc))]

```
git clone --branch multios --single-branch https://github.com/tpc2233/tunet.git
cd tunet

conda create -n tunet python=3.12.9 -y
conda activate tunet

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 

pip install onnx pyyaml lpips onnxruntime Pillow albumentations

```

```
TEST YOUR INSTALLATION:
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Should give your GPU and Torch version, cuda available: True. if not, was not installed properlly. 
```

## For macOS:   
✅ Make sure Miniconda or Anaconda is installed:
###[[Install Video](https://youtu.be/QaAca_LiwKc))]

```
git clone --branch multios --single-branch https://github.com/tpc2233/tunet.git
cd tunet

conda create -n tunet python=3.10 -y
conda activate tunet

pip install torch torchvision torchaudio  

pip install onnx pyyaml lpips onnxruntime Pillow albumentations

```



## For Linux and Multi-GPU use the dedicated Branch:   

```
check branches
```


✅ How to use: 
###[[Training Video](https://youtu.be/gRwQRJPaX7U)] 
```
TRAINING:
1- Choose one of the config_templates for your project (simple or advanced)
2- Set the input plate paths 
You are good to go!
```
 
SINGLE-GPU  
Run the trainer:  
   
- Windows or macOS   
```
python train.py --config /path/to/your/config.yaml
```  


- Linux
```
check branches
```   
   




## Training screen:
```
left: Src Original plate
Middle: Dst Modified plate
Right: Model Inference 
```
![Screenshot 2025-04-06 at 18 37 45](https://github.com/user-attachments/assets/bc4ab4b4-d636-4b7b-9003-aaed1b213d02)




## Inference:
- * Important: Make sure you are using correct inference for your branch OS. Do not mix up 
```
python inference.py --overlap_factor 0.25 --checkpoint /path/to/your/tunet_latest.pth --input_dir /path/to/your/plate --output_dir /path/to/output/folder
```

### Inference converters:
- * Important: Make sure you are using correct converter for your branch OS. Do not mix up

✅ Foundry NUKE Converter:
```
Convert the model to run native inside Nuke, CAT:

python utils/convert_nuke.py --method script --generate_nk --checkpoint_pth /path/to/model/tunet_latest.pth
```
<img width="899" alt="Screenshot 2025-04-06 at 19 31 15" src="https://github.com/user-attachments/assets/e8b4c620-93a3-4f50-8789-09f88326c2b6" />



✅ Autodesk FLAME Converter:
```
WIP TODO
Convert the model to run native inside Flame, ONNX:

python utils/convert_flame.py --use_gpu --checkpoint /path/to/model/tunet_latest.pth
```
<img width="893" alt="Screenshot 2025-04-06 at 19 28 21" src="https://github.com/user-attachments/assets/0eec9a04-eb3b-4e1a-94bb-b23f9d441690" />


## Video:
[[video 🤗](https://youtu.be/UyMq0bsny-A)]


### Citation

Consider cite TUNET in your project.
```
@article{tpo2025tunet,
  title={TuNet},
  author={Thiago Porto},
  year={2025}
}
```

## License

The source code is licensed under the Apache License, Version 2.0.
Commercial use Permission 

