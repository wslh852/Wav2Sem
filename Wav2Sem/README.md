## **Wav2Sem**
**Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial Animation**, ***CVPR 2025***.
Hao Li, Ju Dai, Xin Zhao, Feng Zhou, Junjun Pan, Lei Li


## **Introduction**
Wav2Sem is a plug-and-play semantic module that extracts meaningful audio features to decouple near-homophonic syllables in existing self-supervised speech models. It enhances facial animation expressiveness and improves phoneme recognition accuracy.

## **Environment**
- Linux
- Python 3.6+
- Pytorch 1.9.1
- CUDA 11.1 (GPU with at least 11GB VRAM)

Other necessary packages:
```
pip install -r requirements.txt
```
- ffmpeg
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

IMPORTANT: Please make sure to modify the `site-packages/torch/nn/modules/conv.py` file by commenting out the `self.padding_mode != 'zeros'` line to allow for replicated padding for ConvTranspose1d as shown [here](https://github.com/NVIDIA/tacotron2/issues/182).

## **Dataset Preparation**
### LibriSpeech960
Request the LibriSpeech960 data from [https://www.openslr.org/12]

## **Training**
Training file address ：./Wav2Sem/main/train_wav2sem.py

### **Plug-and-play**
We provide a usage case of Wav2Sem in Codetalker at the project address ./CodeTalker_Wav2Sem

### **Model Weights Path**
Link: https://pan.baidu.com/s/11Kr_RO4e6mrGt5TfKScnZQ?pwd=x99m 
Extraction Code: x99m 
