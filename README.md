<div align="center">
<p align="center">
  <h1>TinyMU: A Compact Audio Language Model for Music Understanding</h1>
</p>
</div>

## Overview
TinyMU is a compact (229M) Music-Language Model
with strong understanding and reasoning abilities. It achieves 82% of SOTA LALM’s performance on the MuChoMusic benchmark, while being 35x smaller. 

<div align="center">
  <img src="resource/Logo.png" alt="" width="700">
</div>

## Environment setup: 
```
conda create --name tinymu python=3.11
conda activate tinymu

cd TinyMU
pip install -r requirements.txt
```

## Quick Start 
Run
```
python demo.py --audio_path ./resource/example1.wav --prompt "Describe the music you hear."
```
This will automatically download the pretrained TinyMU model to the `./ckpt` folder from the [huggingface repo](https://huggingface.co/AndreasXi/TinyMU), and do the infer. 

## Training
### Data Preparation
To train TinyMU, we organize the data into the following format:
```
[
    {
        "file_name": "_Ra1Y6K7nSs_0.000_10.000.wav",
        "input_text": "Caption the music",
        "target_text": "This reggae song features male voices singing the main melody in harmony...",
        "dataset": "MusicCaps",
        "task": "Captioning"
    },
    ...
]
```
Where the ```file_name``` refers to the name of the audio file, ```input_text``` is the instruction provided to TinyMU, and ```target_text``` is the output used for loss calculation. The ```dataset``` denotes the dataset the sample belongs to, and ```task``` specifies the task type for this sample.
An example of the metadata file can be found in ```data/MusicCaps_train.json```.


After setting up the data, we need to setup the training config file. For this, ensure that each entry in `train_audio_dirs` correctly corresponds to the audio files listed in `train_json_files`. Specifically, the `i`-th directory in `train_audio_dirs` should contain all `.wav` files referenced in `train_json_files[i]`.
An example configuration file is located at ```src/config/train_tinymu.yaml```

### Model Training
Run
```
wandb login
bash scripts/train_tinymu.sh
```
to train a model. 

## Inference
Once training is done, run 
```
bash scripts/infer.sh
```
to use the model to generate an output given a prompt and a music clip. You need to set `exp_dir=$split/$exp_name` where `$split` and `$exp_name` are the ones used in training. This will automatically build a model from the config file and load the pre-trained ckpt.


## Evaluation 
TODO 

<!-- ## Acknowledgement -->
 

## Citation
TODO
