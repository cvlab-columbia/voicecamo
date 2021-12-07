# NVC : Code for the paper Real-Time Neural Voice Camouflage

Website of the project in [voicecamo.cs.columbia.edu](https://voicecamo.cs.columbia.edu)

If you use the code, please cite the paper as:



Please install the necessary requirements with: pip install -r requirements.txt

## Data
We work with the [LibriSpeech dataset](https://www.openslr.org/12/) for this project. To run our code, you 
will need to download their wav files and transcriptions. 

Specifically, the **data** directory has to contain:

The path to this directory has to be introduced in the config file. 


## Pretrained models
The pretrained models reported in our paper can be found in the following links:

To run inference, please just use the command: "python run_inference.py"

Extract the models under the `/path/to/your/checkpoints` directory you introduce in 
the checkpoint_dir argument in the config file. 
