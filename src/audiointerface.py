from multiprocessing import Process, Manager
import time as T
import numpy as np
import sounddevice as sd
import librosa
import torch
import torchaudio
from deepspeech_pytorch.model_copy2 import DeepSpeech, HalfSecNetWav
import pdb


class audioInterface:
    sampleRate = 0
    blockSize = 0
    chunkIdx = 0
    windowSize = .02
    windowStride = .01  # Window stride for spectrogram generation (seconds)
    window = 'hamming'
    model = HalfSecNetWav()
    wav = None
    tapStarted = False
    startTime = 0.0

    def __init__(self, sampleRate, blockSize):
        self.sampleRate = sampleRate
        self.blockSize = blockSize

        self.model.eval()

        state_dict = torch.load(
            "clean_notdenoiser_waveform_scalar_True_future_False_future_amt0.0_firstlayer_False_capped_True_power_0.008_lr_0.00015/check/epoch=03-stepstep=19999.00.ckpt")[
            'state_dict']

        curr_state_dict = self.model.state_dict()
        for name, param in state_dict.items():
            if "halfsec" in name:
                newname = name.split("halfsec.")[1]
                if newname not in curr_state_dict:
                    continue
                else:
                    # backwards compatibility for serialized parameters
                    param = param.data
                curr_state_dict[newname].copy_(param)

        self.wav = self.load_audio(
            "cf057df2-81c9-4dde-8c3c-cf42ca73ee1a_input_batch_4_iter_1.wav")

    def load_audio(self, path):
        sound, sample_rate = torchaudio.load(path)
        if sound.shape[0] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=0)  # multiple channels, average
        return sound.numpy()

    def prepareAudio(self):

        # y = y.reshape(-1)
        y = self.wav
        # print(y.shape)
        # y = librosa.resample(y, 48000, 16000)
        # print(y.shape)
        n_fft = int(self.sampleRate * self.windowSize)
        win_length = n_fft
        hop_length = int(self.sampleRate * self.windowStride)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=self.window)

        real = torch.unsqueeze(torch.FloatTensor(np.real(D)), dim=0)
        imag = torch.unsqueeze(torch.FloatTensor(np.imag(D)), dim=0)
        inputs = torch.cat([real, imag], dim=0)

        return torch.unsqueeze(inputs[:, :, :201], dim=0)

    def infer(self):

        modelInput = self.prepareAudio(input)

        modelOutput = self.model(modelInput)

        # npOut = librosa.resample(modelOutput.detach().numpy().reshape(-1), 16000, 48000)
        pdb.set_trace()
        npOut = modelOutput.detach().numpy().reshape(-1)
        # modelOutput = np.random.rand(int(self.sampleRate*0.5))
        return np.split(npOut, int((self.sampleRate / self.blockSize) * 0.5))

    def nextAudioChunk(self, inputArrays):
        # print("len input arrays: {}".format(len(inputArrays)))
        # print("chunkidx: {}".format(self.chunkIdx))

        if self.chunkIdx + int(self.sampleRate / self.blockSize) * 2 < len(inputArrays) - 1:
            print("in next audio chunk ")

            out = np.concatenate(inputArrays[self.chunkIdx:self.chunkIdx + int(self.sampleRate / self.blockSize) * 2])
            print(out)

            self.chunkIdx += int((self.sampleRate / self.blockSize) * 0.5)

            return out

        # else: return np.zeros((int(self.sampleRate/self.blockSize)*2,1))
        else:
            return None

    def runNN(self, input, output):
        """
        run inference
        """
        while True:

            # print("current length of output q: {}".format(len(output.queue)))

            if len(input) > int((self.sampleRate / self.blockSize) * 3):

                nextInput = self.nextAudioChunk(inputArrays=input)

                if nextInput is not None:

                    nextOutput = self.infer(nextInput)
                    #nextOutput = np.split(nextInput, int((self.sampleRate / self.blockSize) * 2))

                    #n_out = nextOutput[:2]


                    for i in nextOutput:
                        output.put(i.reshape(-1, 1))

    def runAudioStream(self, inputArrays, outputQueue):
        """
        run the audio stream
        """

        def callback(indata, outdata, frames, time, status):
            """
            for 2 seconds of audio, (sr/frames)*2

            """

            nonlocal self
            if not self.tapStarted:
                print("Tap Started")
                self.startTime = time.inputBufferAdcTime
                self.tapStarted = True

            print(time.inputBufferAdcTime - self.startTime)

            nonlocal inputArrays
            nonlocal outputQueue

            inputArrays.append(indata)

            if not outputQueue.empty():

                print("output q has data")
                outdata[:] = outputQueue.get()
            else:
                print("output queue empty!")
                outdata[:] = np.zeros_like(outdata)
                # outdata[:] = np.random.randn(*outdata.shape)

        with sd.Stream(samplerate=self.sampleRate, blocksize=self.blockSize, channels=1, callback=callback):
            while True:
                continue

    def run(self):

        with Manager() as manager:
            inputArrays = manager.list()

            outputQueue = manager.Queue()

            #
            for i in range(3*int(self.sampleRate/self.blockSize)):
                outputQueue.put(np.zeros((self.blockSize,1)))

            # creating processes
            p1 = Process(target=self.runNN, args=(inputArrays, outputQueue))
            p2 = Process(target=self.runAudioStream, args=(inputArrays, outputQueue))

            # starting process 1
            p1.start()
            # starting process 2
            p2.start()

            # wait until process 1 is finished
            p1.join()
            # wait until process 2 is finished
            p2.join()

            # both processes finished
            print("Done!")


if __name__ == '__main__':
    audio = audioInterface(sampleRate=16000, blockSize=4000)
    audio.infer()

    #audio.run()
