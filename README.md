<div align="center">

# TUNET
A direct, pixel-level mapping from src to dst images via encoder-decoder.   
Supports, training, inference or export to Compositor tools such Foundry Nuke or Autodesk Flame native inference.


</div>

## 🎥 Showcases: 
AOVs:   
![540_ezgif-45f24ebccb11e0](https://github.com/user-attachments/assets/507167b0-c473-44c0-b444-55f48bc7843b)   
https://youtu.be/TwvN8axWJLY   
Models from the video are available for Nuke and Flame can be downloaded link below for test locally:   
https://f.io/HovatFeX   
   
Models has been trained in combination with Nvidia Cosmos foundation models.   
Using 8x B200 GPUs. Inference can be done in consumer GPUs.



Rain:   
Flame:   
[![Flame video](https://img.youtube.com/vi/6-OFAJtfliM/hqdefault.jpg)](https://youtu.be/6-OFAJtfliM)


## For Windows:   
✅ Make sure Miniconda or Anaconda is installed:
###[[Install Video](https://youtu.be/QaAca_LiwKc))]

```
git clone --branch multios --single-branch https://github.com/tpc2233/tunet.git
cd tunet

conda create -n tunet python=3.12.9 -y
conda activate tunet

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 

pip install onnx pyyaml lpips onnxruntime Pillow albumentations PySide6

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

pip install onnx pyyaml lpips onnxruntime Pillow albumentations PySide6

```



## For Linux and Multi-GPU use the dedicated Branch:   

```
check branches
```


✅ How to use, Open Tunet UI: 
###[[Training Video soon]] 
```
python ui_app.py

You are good to go!
```


✅ Tunet UI:
## Main:
![main](https://github.com/user-attachments/assets/fe8f03e1-2e53-46b9-8c91-b85befdb3ff9)

## Training:
![train5](https://github.com/user-attachments/assets/f6bf9ba3-f84b-4d0d-9d95-ff418bebaaf5)

## Converter:
![convert](https://github.com/user-attachments/assets/8633b5ff-0152-4a09-afaf-1fa7e4669359)

## Adv:
![adv](https://github.com/user-attachments/assets/f27df935-6677-4c8f-adfb-dbc3b7cd8c9d)

## Data control:
![data](https://github.com/user-attachments/assets/a30e0160-dc23-4623-9106-d3237700be94)



### Terminal CMD:

✅ Command line and terminal trainings are still available as usual, check docs:
```
soon
```


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

