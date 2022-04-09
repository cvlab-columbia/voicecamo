# NVC : Code for the paper Real-Time Neural Voice Camouflage

Website of the project in [voicecamo.cs.columbia.edu](https://voicecamo.cs.columbia.edu)

Please install the necessary requirements with: pip install -r requirements.txt

## Data
We work with the [LibriSpeech dataset](https://www.openslr.org/12/) for this project. To run our code, you 
will need to download their wav files and transcriptions. You can do this by running the data/librispeech.py file. This will generate the manifest json files. If this step was done correctly, the LibriSpeech dataset will be a folder in the data folder as such: data/LibriSpeech_dataset. This folder will contain test_clean, train, and val folders. Within each of these folders you should have a txt and a wav folder. 

The path to this directory has to be introduced in the config file. 


## Pretrained models
The pretrained models reported in our paper can be found by emailing the author at mac2500@columbia.edu, at which point we will grant you access to our google drive link with the checkpoints.
Extract the models under the `/path/to/your/checkpoints` directory you introduce in
the checkpoint_dir argument in the config file.

## Inference
To run inference, please just use the command: "python run_inference.py"

## Training
To train your own modle, please run the command: "python run_train.py"

