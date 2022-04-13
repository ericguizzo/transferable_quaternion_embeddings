# Learning Speech Emotion Representations in the Quaternion Domain
This repository supports the [paper](https://arxiv.org/abs/2204.02385) "Learning Speech Emotion Representations in the Quaternion Domain" submitted to IEEE Transactions of Audio, Speech and Language processing. Here you can find easy instructions for the download of the required data  and our pre-trained weights, for training from scratch RH-emo on Iemocap and for the application of our approach to a generic speech emotion recognition dataset.


## Installation
Our code is based on Python 3.7.

To install all required dependencies run:
```bash
pip install -r requirements.txt
```


## Data download and preprocessing
* Follow these instructions to download the Iemocap dataset: https://sail.usc.edu/iemocap/
* Put the path to the downloaded dataset in the *input_iemocap_folder* variable in the *preprocessing_config.ini* file.
* Run the following command to pre-process the dataset:
```bash
python3 preprocessing.py
```

It is possible to download our pre-trained RH-emo weights with this command:
```bash
python3 download_weights.py
```
These weights are also available for manual download [here](https://drive.google.com/file/d/1vCX0KHW44Q9plKTdkgyKZRcyjfgVA7jX/view?usp=sharing).

If you use our pretrained weights skip the following section.


## RH-emo pretraining
Once downloaded and preprocessed Iemocap it is possible to run the RH-emo pretraining from scratch with this command:
```bash
python3 exp_instance.py --ids [1] --last 2 --gpu_id 0
```
This script will run the training *training_RHemo.py* with our best hyperparameters, which are specified in the configuration file *experiments/1_RHemo_train_onlyrecon.txt*.
Two consecutive trainings are launched: without and with the emotion classification term in the loss function, as explained in the paper. When the trainings finish, a metrics spreadsheet is saved in the *results* folder. The results will match the ones exposed in the original paper.


## Method application
With a pretrained RH-emo network it is possible to use quaternion-valued networks for speech emotion recognition starting from monoaural spectrograms. It is sufficient to call the function ```get_embeddings()``` on a pretrained RH-emo as a preprocessing step before the forward propagation. We provide quaternion implementations of the *AlexNet*, *ResNet50* and *VGG16* networks.
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

You can run our speech emotion recognition training on Iemocap with this command:
```bash
python3 exp_instance.py --ids [2] --last 3 --gpu_id 0
```
The script will launch 3 consecutive trainings using the quaternion AlexNet, ResNet50 and VGG16 with Iemocap and will return a metrics spreadsheet that can be found in the *results* folder. The results will match the ones exposed in the original paper.
