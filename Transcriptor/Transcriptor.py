import numpy as np
import pandas as pd
from tqdm import tqdm
import whisper
import stable_whisper
from pydub import AudioSegment
# Extracción WAV
import subprocess
import librosa
import soundfile as sf

import subprocess
# MP4 A WAV
def create_wav(video_path,audio_path):
    command = "ffmpeg -i " + video_path + " -ab 160k -ac 2 -ar 44100 -vn " + audio_path 
    subprocess.call(command, shell=True) 

def transcribir(audio_path, model):
    result =model.transcribe('audio2.wav', language="es",suppress_silence=True, ts_num=16)
    result.save_as_json(audio_path.replace('.wav','')+'.json')
    return

