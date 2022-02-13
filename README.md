# Transferable Quaternion Embeddings for Speech Emotion Recognition
This repository supports the [paper](arxiv_link) "Transferable Quaternion Embeddings for Speech Emotion Recognition" submitted to IEEE Transactions of Audio, Speech and Language processing. Here you can find easy instructions for the download of the required data  and our pre-trained weights, for training from scratch R2Hemo on Iemocap and for the application of our approach to a generic speech emotion recognition dataset.


## Installation
Our code is based on Python 3.7.

To install all required dependencies run:
```bash
pip install -r requirements.txt
```


## Data download and preprocessing
* Follow these instructions to download the Iemocap dataset: https://sail.usc.edu/iemocap/
* Run the following scripts to pre-process the dataset:
```bash
python3 preprocessing_IEMOCAP_vad.py
python3 preprocessing_IEMOCAP.py
```

It is possible to download our pre-trained R2Hemo weights with this command:
```bash
python3 download_weights.py --output_path models/
```
These weights are also available for manual download [here](https://drive.google.com/file/d/1vCX0KHW44Q9plKTdkgyKZRcyjfgVA7jX/view?usp=sharing).

If you use our pretrained skip the following section.


## R2Hemo pretraining
Once downloaded and preprocessed Iemocap it is possible to run the R2Hemo pretraining from scratch with this command:
```bash
python3 exp_instance.py --ids [1] --gpu_id 0
```
This script will run the training *training_R2Hemo.py* with the hyperparameters specified in the configuration file *experiments/1_R2Hemo_train_onlyrecon.txt*.
Two consecutive trainings are launched: without and with the emotion classification term in the loss function. When the trainings finishes, a metrics spreadsheet is saved in *path_to_spreadsheet*. The results will match the ones exposed in the original paper.


## Method application
With a pretrained R2Hemo network it is possible to use quaternion-valued networks for speech emotion recognition starting from monoaural spectrograms. It is sufficient to call the function ```get_embeddings()``` on a pretrained R2Hemo as a preprocessing step before the forward propagation. We provide quaternion implementations of the *AlexNet*, *ResNet50* and *VGG16* networks.
An example in pseudocode:
```python3
import torch
from models import *

quaternion_processing = True

model = resnet50(quat=quaternion_processing)
if quaternion_processing:
    r2he = simple_autoencoder_2_vad()
    r2he.load_state_dict(pretrained_dict_r2he, strict=False)

for e in epochs:
  for i, (sounds, truth) in enumerate(dataloader):
        optimizer.zero_grad()
        if quaternion_processing:
            with torch.no_grad():
                sounds, _, _, _, _ = r2he.get_embeddings(sounds)
        pred = model(sounds)
        loss = loss_function(pred, truth)
        loss.backward()
        optimizer.step()
```

You can run our speech emotion recognition training on Iemocap with this line:
```bash
python3 exp_instance.py --ids [2] --gpu_id 0 --last 3
```
The script will launch 3 consecutive trainings using the quaternionAlexNet, ResNet50 and VGG16 with Iemocap and will return a metrics spreadsheet that can be fount at *path_to_spreadsheet*.
