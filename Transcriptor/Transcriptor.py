import numpy as np
import pandas as pd
from tqdm import tqdm
import whisper
from pydub import AudioSegment


def transcribir(audio_path, model, fp = False):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(fp16 = fp, language="es" )
    result = whisper.decode(model, mel, options)
    return result.text

def Trans_Alpha(tiempo, Inicio, Final, audio_path, audio_id, model):
  #Entrega un dataframe con las transcripciones del intervalo de "Inicio" a "Final" divida en segmentos de largo "Tiempo"
    #Se importa audio
    audio = AudioSegment.from_file(audio_path)
    #Se define duracion total
    Final = min(audio.duration_seconds, Final)
    #Cantidad de intervalos
    intervalos = int((Final - Inicio) / tiempo)
    #Larges corresponde a la lista de duraciones de cada segmento
    Larges = [tiempo] * intervalos
    #Si la duracion total no es multiplo de "tiempo" sobrará un segmento menor a "tiempo" que se añade 
    F = Final - np.asarray(Larges).sum() #Si la suma de todos los tiempos es menor al final entonces 
    if F !=0:
        Larges.append(F)                     #agregamos el último intervalo a la lista Larges
    #Lista de transcripciones
    Trans = []
    #Tiempo de inicio
    a = Inicio
    #Lista de inicios de cada segmento
    Inicio = []
    #Lista de Tiempos de cada segmento = Larges
    Tiempo = []
#     for i in tqdm(range(len(Larges))):
    for i in range(len(Larges)):
        Tiempo.append(np.around(Larges[i], 2))
        audio_segment = audio[a*1000:(a + Larges[i]) * 1000]
        audio_segment.export("temp.wav", format="wav")
        result = transcribir("temp.wav",model)
        Trans.append(result)
        if i == 0:
            Inicio.append(np.around(0, 2))
        else:
            Inicio.append(np.around(Larges[i - 1], 2))
        a = Larges[i] + a
    Data_Frame = pd.DataFrame({'Frase': Trans, 'Tiempo': Tiempo, 'Tiempo-1': Inicio})
    Data_Frame['Final'] = Data_Frame['Tiempo'].cumsum()
    Data_Frame['Inicio'] =  Data_Frame['Tiempo-1'].cumsum()
    Data_Frame['clase_id'] = audio_id
    Data_Frame['window'] = 1
    n = 1
    for i in range(len(Data_Frame['clase_id'])):
        Data_Frame['window'].loc[i] = n
        n = n + 1
    df = Data_Frame[['clase_id', 'window', 'Inicio', 'Final', 'Tiempo', 'Frase']]

    return df

def data_transcriptor(df, model, audio_path):
    #Recibe los intervalos de tiempos a transcribir
    Transcripcion = pd.DataFrame()
    for index, row in df.iterrows():
        audio_id = row['id']
        Inicio = row['Inicio']
        Final = row['Final']
        Tiempo = Final - Inicio
        Data = Trans_Alpha(tiempo = Tiempo, Inicio=Inicio, Final=Final  , audio_path = audio_path, audio_id = audio_id, model =model)
        if 'Hablante' in df.columns:
            Data['Hablante'] = row['Hablante']
        Transcripcion = pd.concat([Transcripcion, Data])
    Transcripcion = Transcripcion.sort_values(by=['Inicio']).reset_index(drop=True)
    return Transcripcion
