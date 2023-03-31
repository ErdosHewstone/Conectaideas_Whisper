import subprocess
import stable_whisper
import json
import pandas as pd
from pydub import AudioSegment
import math
from tqdm import tqdm

def segmentos_de_tiempo(audio_path):
    # Carga el archivo de audio
    audio = AudioSegment.from_wav(audio_path)
    audio_length = len(audio)
    if audio_length > 17*60*1000:

        # Define la duración máxima de cada segmento
        m = 17
        s = 60
        x = m*s*1000

        # Busca la duración máxima de los segmentos
        while audio_length % x != 0:
            if s > 1:
                s -= 1
            else:
                s = 60
                m -= 1
            x = m*s*1000
            if m <= 12:
                break

        # Divide el audio en segmentos
        if m > 12:
            num_segments = audio_length // x
            segment_length = x
            segment_times = [(i*segment_length, (i+1)*segment_length) for i in range(num_segments)]
        else:
            segment_length = 17*60*1000
            num_segments = math.ceil(audio_length / segment_length)
            last_segment_length = audio_length - (num_segments-1)*segment_length
            segment_times = [(i*segment_length, (i+1)*segment_length) for i in range(num_segments-1)]
            segment_times.append(((num_segments-1)*segment_length, (num_segments-1)*segment_length+last_segment_length))
    else:
        return [(0,audio_length)]
    # Retorna los tiempos de inicio y fin de cada segmento
    return segment_times
# MP4 A WAV
def create_wav(video_path,audio_path):
    command = "ffmpeg -i " + video_path + " -ab 160k -ac 2 -ar 44100 -vn " + audio_path 
    subprocess.call(command, shell=True) 

# Transcripción de audio
def transcribir(audio_path, model, tiempos = []):
    if len(tiempos)>0:
        temps = []
        for t in tiempos:
            t1 = t[0]*1000
            t2 = t[1]*1000
            temps.append([t1,t2])
        tiempos = temps
    else:
        tiempos = segmentos_de_tiempo(audio_path)
    newAudio = AudioSegment.from_wav(audio_path)
    results = []
    for t1,t2 in tqdm(tiempos):
        a = newAudio[t1:t2]
        a.export("temp.wav", format="wav") 
        result = model.transcribe("temp.wav", language="es", suppress_silence=True, ts_num=16)
        results.append(result.to_dict())
    return results, audio_path.replace('.wav','')
#     return datos, audio_path.replace('.wav','')

def json_to_dataframe(result):
    idx = result[1]
    results = result[0]
    df = pd.DataFrame()
    for result in results:
        segments = result['segments']
        print(result)
        data = []
        last_end = 0
        for segment in segments:
            words = segment['words']
            words_list = [word['word'] for word in words]
            start_list = [word['start'] for word in words]
            end_list = [word['end'] for word in words]
            data.append({
                'text': segment['text'],
                'start': last_end + segment['start'],
                'end': last_end + segment['end'],
                'avg_logprob' : segment['avg_logprob'],
                'no_speech_prob': segment['no_speech_prob'],
                'words': words_list,
                'words_start': start_list,
                'words_end': end_list,
                'words_len': len(words_list)
            })
        data = pd.DataFrame(data)
        last_end = data.iloc[-1]['end']
        df = pd.concat([df, data])
    df['id'] = idx
    df.index = range(len(df))
    return df

def run_model(modelo):
    return stable_whisper.load_model(modelo)

# Función de envoltura principal
def main():
    video_path = "ruta/al/video.mp4"
    audio_path = "ruta/al/audio.wav"
    modelo = "base"
    model = stable_whisper.load_model(modelo)
    result = {}
    create_wav(video_path, audio_path)
    transcribir(audio_path, model)
    run_model(modelo)
    json_to_dataframe(result)
# Si este archivo se ejecuta como el archivo principal, llame a la función de envoltura principal
if __name__ == "__main__":
    main()
