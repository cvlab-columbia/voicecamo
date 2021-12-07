import json
import torch
import Levenshtein as Lev
import soundfile
import matplotlib.pyplot as plt
import os
import librosa
import numpy as np
import pdb
import uuid


from omegaconf.dictconfig import DictConfig
from src.loader.data_loader import SpectrogramParser
from hydra import utils
from librosa import display
from src.loader.data_module import DeepSpeechDataModule
from src.model import DeepSpeech
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything
from src.decoder import Decoder, BeamCTCDecoder
from src.decoder import GreedyDecoder
from torch.cuda.amp import autocast
#from src.visualizations import fast_istft
#from ctcdecode import CTCBeamDecoder
torch.multiprocessing.set_sharing_strategy('file_system')


def get_stft(y):
    sample_rate=16000
    window_size=.02
    window_stride=.01
    window='hamming'
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    real = torch.FloatTensor(np.real(D))
    imag = torch.FloatTensor(np.imag(D))

    return [real, imag]

def griffin_convert_mag_to_audio(mag_noise, audio_path):

    mag_noise = mag_noise.detach().cpu().numpy()
    y_inv = librosa.griffinlim(mag_noise)

    return y_inv

def run_transcribe(audio_path: str,
                   spect_parser: SpectrogramParser,
                   model: DeepSpeech,
                   decoder: Decoder,
                   device: torch.device,
                   precision: int,
                   mag_noise: torch.tensor = None):
    real, imag, y = spect_parser.parse_audio(audio_path)
    inputs = torch.zeros(1, 2, real.shape[0], real.shape[1])
    inputs[0,0,:,:]=real
    inputs[0,1,:,:] = imag
    spect = inputs.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int().to(device)

    with autocast(enabled=precision == 16):

        out, output_sizes, actual_noise, mag_noise, mag_input= model(spect, input_sizes)

    y_inv = griffin_convert_mag_to_audio(mag_noise[0], audio_path)
    y_inv_input = griffin_convert_mag_to_audio(mag_input[0] + mag_noise[0], audio_path)

    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    pdb.set_trace()
    return decoded_output, decoded_offsets


def wer_calc(s1, s2):


    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance(''.join(w1), ''.join(w2))

    """Compare WERs"""
def power_law(data, power=0.3):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data >= 0] = 1
    mask[data < 0] = -1
    data = np.power(np.abs(data), power)
    data = data * mask
    return data

N_FFT = 320
HOP_LENGTH = 160
WIN_LENGTH = 320

SIGMOID_THRESHOLD = 0.5

def fast_istft(F, power=False, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    # directly transform the frequency domain data to time domain data
    # apply power law
    T = torch.istft(F, n_fft=320, win_length=320, hop_length=int(16000 * 0.01))
    if power:
        T = power_law(T, (1.0 / 0.3))
    return T

def cer_calc(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def log_audio_specs_trans_noswap(power,path, num,input_sizes,index, inputs, actual_noise, mag_input, mag_random, mag_noise, gt_transcript,pred_asr, pred_random_asr, pred_transcript, wer_clean, wer_random, wer):
    j = num

    inputs = inputs.detach().clone().cpu()
    actual_noise = actual_noise.detach().clone().cpu()
    mag_noise = mag_noise.detach().clone().cpu().numpy()
    mag_input = mag_input.detach().clone().cpu().numpy()

    input_sig = fast_istft((inputs[:, 0, :, :] + inputs[:, 1, :, :] * 1j)[j]).numpy()
    noise_sig = fast_istft((actual_noise[:, 0, :, :] + actual_noise[:, 1, :, :] * 1j)[j]).numpy()


    if not os.path.isdir(path):
        os.mkdir(path)
    uuidv = uuid.uuid4()
    soundfile.write(path + "/inputs/" + str(uuidv) + "_input_batch_" + str(index) + "_iter_" + str(num) + ".wav",
                    input_sig, 16000)

    soundfile.write(path + "/noise/" + str(uuidv) + "_noise_batch_" + str(index) + "_iter_" + str(num) + ".wav",
                    noise_sig, 16000)

    soundfile.write(path + "/added/" + str(uuidv) + "_added_batch_" + str(index) + "_iter_" + str(num) + ".wav",
                    input_sig + noise_sig, 16000)

    with open(path + "/trans/" + str(uuidv) + "_trans_batch_" + str(index) + "_iter_" + str(num), 'w') as f:
        f.write(gt_transcript)

    return

"""Here we want to let the model predict noise, not give it, 
but then check the size of that noise, and pass in white noise of the max of that magnitude."""
def evaluate(cfg: DictConfig):

    cfg.wandb = False
    os.chdir(utils.get_original_cwd())

    seed_everything(cfg.SEED)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)


    if cfg.trainer.accelerator == "ddp":
        is_distributed=True
    else:
        is_distributed=False

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        is_distributed=is_distributed
    )

    #list_of_checks=["../../../deepspeechattack/deepspeech.pytorch/checkpoints/clean_notdenoiser_waveform_scalar_True_future_True_future_amt50_firstlayer_False_capped_True_power_0.008_lr_0.00015/check/epoch=02-stepstep=15299.00.ckpt"]

    model = DeepSpeech(
        wandb=cfg.wandb,
        future=cfg.future,
        future_amt=50,
        residual=cfg.residual,
        batchnorm=cfg.batchnorm,
        waveform=cfg.waveform,
        firstlayer=cfg.firstlayer,
        capped=cfg.capped,
        inputreal=cfg.inputreal,
        power=cfg.power,
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )

    device = 'cuda:0'
    model = model.to(device)

    model.load_state_dict(torch.load("../../../deepspeechattack/deepspeech.pytorch/checkpoints/clean_notdenoiser_waveform_scalar_True_future_True_future_amt50_firstlayer_False_capped_True_power_0.008_lr_0.00015/check/epoch=02-stepstep=15299.00.ckpt")['state_dict'], strict=False)

    model.eval()

    test_loader = data_loader.test_dataloader()
    list_of_lev_dist=[]
    list_of_clean_lev_dist=[]
    list_of_n_words=[]
    list_of_n_chars=[]
    list_of_cer_dist=[]
    list_of_clean_cer_dist=[]
    list_of_lev_dist_random=[]

    with torch.no_grad():
        decoder = GreedyDecoder(labels)
        """decoder = BeamCTCDecoder(labels, lm_path="4-gram.arpa",
                                 alpha=0,
                                 beta=0,
                                 cutoff_top_n=40,
                                 cutoff_prob=1.0,
                                 beam_width=100,
                                 num_processes=4,
                                 blank_index=0
                                 )"""

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx > 1:

                print(batch_idx, " / ", len(test_loader))

                inputs, targets, mag_noises, input_percentages, target_sizes, scalar = batch
                inputs = inputs.to(device)
                input_percentages = input_percentages.to(device)
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                output = model(inputs, input_sizes, scalar)

                if output is not None:

                    x, output_lengths, actual_noise, actual_noise_scaled, mag_noise, mag_input, max_y = output
                    #print(max_y)
                    mag_input = torch.sqrt(inputs[:, 0, :, :] ** 2 + inputs[:, 1, :, :] ** 2)

                    lengths = input_sizes.cpu().int()
                    output_lengths = model.get_seq_lens(lengths)

                    xclean = model.run_through_full_network(mag_input, torch.zeros(mag_input.shape).cuda(), output_lengths)
                    actual_noise = actual_noise.cpu().detach()
                    noise_sig = fast_istft(
                        (actual_noise[:, 0, :, :] + actual_noise[:, 1, :, :] * 1j)[0]).numpy()
                    amt=0.4
                    random_sig = np.random.uniform(-amt*0.05, amt*0.05, size=noise_sig.shape)
                    D = librosa.stft(random_sig, n_fft=320, hop_length=160, win_length=320, window="hamming")
                    real = torch.unsqueeze(torch.FloatTensor(np.real(D)), dim=0)
                    imag = torch.unsqueeze(torch.FloatTensor(np.imag(D)), dim=0)
                    inputs_noise = torch.cat([real, imag], dim=0)

                    random_noise = torch.sqrt(inputs_noise[0, :, :] ** 2 + inputs_noise[1, :, :] ** 2)
                    random_noise = torch.unsqueeze(random_noise, dim=0)
                    random_noise = random_noise.repeat(actual_noise.shape[0], 1, 1)

                    xrandom = model.run_through_full_network(mag_input, random_noise.cuda(), output_lengths)#torch.unsqueeze(output,dim=0).repeat(4,1,1).cuda()

                    """outclean = xclean.transpose(0,1)
                    outclean = outclean.log_softmax(-1)
                    out = x.transpose(0, 1)  # TxNxH
                    out = out.log_softmax(-1)"""
                    """loss_ctc = self.criterion(out, targets, output_lengths, target_sizes)"""


                    with torch.no_grad():
                        decoded_output, _ = decoder.decode(x, output_lengths)
                        decoded_output_clean, _ = decoder.decode(xclean, output_lengths)

                        decoded_output_random, _ = decoder.decode(xrandom, output_lengths)


                        d_target = [labels[targets[i].item()] for i in range(len(targets))]
                        sum=0
                        for k in range(inputs.shape[0]):

                            d_target = [labels[targets[i].item()] for i in range(len(targets))]
                            decoded_target = ""

                            decoded_target = decoded_target.join(d_target[sum:sum+target_sizes[k]])
                            sum = sum + target_sizes[k]

                            lev_dist = wer_calc(decoded_target, decoded_output[k][0])
                            lev_dist_clean = wer_calc(decoded_target, decoded_output_clean[k][0])

                            if len(decoded_target.split()) > 0:

                                decoded_output_random, _ = decoder.decode(xrandom, output_lengths)
                                lev_dist_random = wer_calc(decoded_target, decoded_output_random[k][0])

                                nwords = len(decoded_target.split())
                                nchars = len(decoded_target.replace(' ', ''))
                                list_of_lev_dist.append(lev_dist)
                                list_of_clean_lev_dist.append(lev_dist_clean)
                                list_of_n_words.append(nwords)
                                list_of_n_chars.append(nchars)

                                cer_dist = cer_calc(decoded_target, decoded_output[k][0])
                                cer_dist_clean = cer_calc(decoded_target, decoded_output_clean[k][0])
                                list_of_cer_dist.append(cer_dist)
                                list_of_clean_cer_dist.append(cer_dist_clean)
                                werattack = lev_dist / nwords
                                werclean = lev_dist_clean / nwords
                                werrandom = lev_dist_clean / nwords

                                list_of_lev_dist_random.append(lev_dist_random)

                                print("wer clean: " ,werclean, "wer attacked: " ,werattack)

                                """if not os.path.isdir("savingattacks"):
                                    os.mkdir("savingattacks")
                                log_audio_specs_trans_noswap(0.5, "savingattacks", k, input_sizes,
                                                      batch_idx, inputs,
                                                      actual_noise.cpu().detach(), mag_input, random_noise, mag_noise,
                                                      decoded_target, decoded_output_clean[k][0],decoded_output_random[k][0],
                                                      decoded_output[k][0],
                                                      werclean, werrandom, wer1)"""

                else:
                    print("None")


        wer = np.mean(np.sum(np.array(list_of_lev_dist)) / np.sum(np.array(list_of_n_words)))
        wer_random = np.mean(np.sum(np.array(list_of_lev_dist_random)) / np.sum(np.array(list_of_n_words)))
        cer_random = np.mean(np.sum(np.array(list_of_lev_dist_random)) / np.sum(np.array(list_of_n_chars)))
        clean_wer = np.mean(np.sum(np.array(list_of_clean_lev_dist)) / np.sum(np.array(list_of_n_words)))
        cer = np.mean(np.sum(np.array(list_of_cer_dist)) / np.sum(np.array(list_of_n_chars)))
        clean_cer = np.mean(np.sum(np.array(list_of_clean_cer_dist)) / np.sum(np.array(list_of_n_chars)))


        with open("results.txt", "a") as myfile:
            myfile.write("wer: " + str(wer) + "\n")
            myfile.write("cer: " + str(cer) + "\n")
            myfile.write("clean wer: " + str(clean_wer) + "\n")
            myfile.write("clean cer: " + str(clean_cer) + "\n")

        print(wer)
        print(cer)
        print(clean_wer)
        print(clean_cer)
