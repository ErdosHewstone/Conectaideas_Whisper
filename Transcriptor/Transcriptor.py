import subprocess
import stable_whisper
import json
import pandas as pd
# MP4 A WAV
def create_wav(video_path,audio_path):
    command = "ffmpeg -i " + video_path + " -ab 160k -ac 2 -ar 44100 -vn " + audio_path 
    subprocess.call(command, shell=True) 

# Transcripción de audio
def transcribir(audio_path, model):
    result = model.transcribe(audio_path, language="es", suppress_silence=True, ts_num=16)
    return result.to_dict(), audio_path.replace('.wav','')
#     return datos, audio_path.replace('.wav','')

def json_to_dataframe(result):
    idx = result[1]
    result = result[0]
    segments = result['segments']
    data = []
    for segment in segments:
        words = segment['words']
        words_list = [word['word'] for word in words]
        start_list = [word['start'] for word in words]
        end_list = [word['end'] for word in words]
        data.append({
            'text': segment['text'],
            'start': segment['start'],
            'end': segment['end'],
            'avg_logprob' : segment['avg_logprob'],
            'no_speech_prob': segment['no_speech_prob'],
            'words': words_list,
            'words_start': start_list,
            'words_end': end_list,
            'words_len': len(words_list)
        })
    df = pd.DataFrame(data)
    df['id'] = idx
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
