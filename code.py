
import deepspeech
import numpy as np
import os
import wave
from pydub import AudioSegment

#convert flac audio to wav using pydub
def flac_to_wav(input_path):
    destination = os.path.splitext(input_path)[0]+'.wav'
    sound= AudioSegment.from_file(input_path, format='flac')
    sound.export(destination, format='wav')
    return destination

#read wav audio file and extract sample rate and frame rate
def read_wav_file(filename) -> [bytes, int]:
   
    with wave.open(flac_to_wav(filename), 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)

    return buffer, rate


#load model
model_file_path = './deepspeech-0.6.0-models/output_graph.pbmm'
beam_width = 500
model = deepspeech.Model(model_file_path, beam_width)

lm_file_path = './deepspeech-0.6.0-models/lm.binary'
trie_file_path = './deepspeech-0.6.0-models/trie'
lm_alpha = 0.75
lm_beta = 1.85
model.enableDecoderWithLM(lm_file_path, trie_file_path, lm_alpha, lm_beta)


#use model to transcribe text
def deepspeech_text(filename: str, lang: str) -> str:
    buffer, rate = read_wav_file(filename)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    return model.stt(data16)

#list all audio files from libriSpeech dataset
directory ='./LibriSpeech/test-other'
for paths, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".flac"):
            print('\nReading file: "{}"'.format(os.path.join(paths,file)))
            print('deepspeech-text: "{}"\n'.format(
          deepspeech_text(os.path.join(paths,file),'en-US')))

