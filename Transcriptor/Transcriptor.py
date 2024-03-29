import subprocess
import stable_whisper
import json
import pandas as pd
from pydub import AudioSegment
import math
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import librosa

def process_audio(wav_path): #ENTREGA EL DATAFRAME DE LOS AUDIOFEATURES PARA CLASIFICAR
    audio = AudioSegment.from_wav(wav_path)
    rate_audio = audio.frame_rate
    segment_length_ms = 10000  # Duración de cada segmento en milisegundos
    num_segments = len(audio) // segment_length_ms
    
    dataframes = []

    for ind in tqdm(range(num_segments)):
        segment = audio[ind * segment_length_ms : (ind + 1) * segment_length_ms]
        segment_samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        
        n_fft = 2048
        hop_length = 512
        S = np.abs(librosa.stft(segment_samples, n_fft=n_fft, hop_length=hop_length)) ** 2
        n_mels = 128
        mel_spec = librosa.feature.melspectrogram(S=S, sr=rate_audio, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec)
        mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(S=log_mel_spec, sr=rate_audio)
        pitches, magnitudes = librosa.core.piptrack(S=log_mel_spec, sr=rate_audio)
        pitch_mean = np.mean(pitches)
        pitch_std = np.std(pitches)
        pitch_confidence = np.mean(magnitudes)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=rate_audio))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=rate_audio))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=rate_audio))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(segment_samples))

        df = pd.DataFrame({
            'mean': [np.mean(np.abs(segment_samples))],
            'std': [np.std(np.abs(segment_samples))],
            'pitch_mean': [pitch_mean],
            'pitch_std': [pitch_std],
            'pitch_confidence': [pitch_confidence],
            'spectral_centroid': [spectral_centroid],
            'spectral_bandwidth': [spectral_bandwidth],
            'spectral_rolloff': [spectral_rolloff],
            'zero_crossing_rate': [zero_crossing_rate]
        })

        for i in range(mfcc.shape[0]):
            df[f'mfcc_{i}'] = np.mean(mfcc[i, :])

        for i in range(spectral_contrast.shape[0]):
            df[f'spectral_contrast_{i}'] = np.mean(spectral_contrast[i, :])

        for q in np.arange(0.1, 1.0, 0.1):
            col_name = f'quantile{int(q * 100)}'
            df[col_name] = np.quantile(np.abs(segment_samples), q)

        dataframes.append(df)

    # Concatenar todos los dataframes de los segmentos en un solo dataframe
    final_df = pd.concat(dataframes, ignore_index=True)
    return final_df


class ClassificationModel_2(nn.Module): #CLASIFICACION: ACEPTABLE, NO ACEPTALE
    def __init__(self):
        super(ClassificationModel_2, self).__init__()
        self.fc1 = nn.Linear(38, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 2)  
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.softmax(self.fc5(x), dim=1)  # Aplicar función de activación softmax en la capa de salida
        return x
    
class ClassificationModel_4(nn.Module): #CLASIFICACION: IDEAL, BUENO, REGULAR, MALA
    def __init__(self):
        super(ClassificationModel_4, self).__init__()
        self.fc1 = nn.Linear(38, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 4)  
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.softmax(self.fc5(x), dim=1)  # Aplicar función de activación softmax en la capa de salida
        return x

#MODELOS
import os as os

modelCND2 = ClassificationModel_2()
model_path = os.path.join(os.path.dirname(__file__), 'modelo_CND2.pth')
modelCND2.load_state_dict(torch.load(model_path))
modelCND2.eval()

modelCND4 = ClassificationModel_4()
model_path = os.path.join(os.path.dirname(__file__), 'modelo_CND4.pth')
modelCND4.load_state_dict(torch.load(model_path))
modelCND4.eval()

def classify_audio_segments(df, model): #Entrega el dataframe de audiofeatures pero con la clasificacion
    scaler = MinMaxScaler()
    df = df.copy()
    # Dividir el dataframe en chunks de tamaño maximo 300
    chunks = np.array_split(df, np.ceil(len(df) / 300))
    datas2 = []
    for chunk in chunks:
        # Crear una lista para almacenar las clasificaciones
        X = chunk
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        # Realizar la clasificación
        with torch.no_grad():
            y_pred = model(X_tensor)
            classification = y_pred.argmax(dim=1).numpy()
        # Agregar la columna de clasificaciones al DataFrame
        chunk['clasificacion'] = classification
        datas2.append(chunk)

    df = pd.concat(datas2)
    return df

def transcripcion_vacia(t1, t2): #FUNCION PARA GENERAR TRANSCRIPCIONES VACIAS DONDE SE SALTA LAS TRANSCRIPCIONES
    result = {'text': "",
        'segments': [
            {
                'text': "",
                'start': 0,
                'end': t2-t1,
                'avg_logprob': 0.0,  
                'no_speech_prob': 1.0,  
                'words': [
                    {
                        'word': "",
                        'start': 0,
                        'end': t2-t1,
                    }
                ]
            }
        ]
    }
    return result

def clasifica_tuplas(tuplas, T1, T2):
    result = [] # Contendrá los resultados

    for i in range(len(T1)): # Iterar sobre T1 y T2
        t1 = T1[i]
        t2 = T2[i]

        total_tiempo = 0
        suma_clasificaciones = 0
        
        for tupla in tuplas:
            # Si hay alguna superposición entre las dos tuplas
            if max(t1, tupla[0]) < min(t2, tupla[1]):
                # Calcular la intersección
                start = max(t1, tupla[0])
                end = min(t2, tupla[1])
                
                # Calcular tiempo en la intersección y añadir a total_tiempo
                tiempo_interseccion = end - start
                total_tiempo += tiempo_interseccion
                
                # Sumar la clasificación ponderada por tiempo
                suma_clasificaciones += tupla[2] * tiempo_interseccion
        
        if total_tiempo > 0: # Prevenir la división por cero
            clasificacion = int(suma_clasificaciones / total_tiempo) # Calcular el promedio de las clasificaciones
            result.append((t1, t2, clasificacion)) # Agregar a los resultados
        else:
            clasificacion = 3 # Calcular el promedio de las clasificaciones
            result.append((t1, t2, clasificacion)) # Agregar a los resultados
    return result

def get_time_intervals(df_classified): #RECIBE LAS CLASIFICACIONES Y JUNTA LOS INTERVALOS DE IGUAL CLASIFICACION
    df_classified = df_classified.copy()
    time_intervals = []
    start_time = 0
    current_classification = df_classified.loc[0, 'clasificacion']

    for index, row in df_classified.iterrows():
        if row['clasificacion'] != current_classification:
            end_time = index * 10  # Cada segmento tiene una duración de 10 segundos
            time_intervals.append((start_time, end_time, current_classification))
            start_time = end_time
            current_classification = row['clasificacion']

    # Agregar el último intervalo de tiempo
    end_time = len(df_classified) * 10
    time_intervals.append((start_time, end_time, current_classification))

    return time_intervals


def dividir_segmentos(t_1, t_2, clasificacion): #DIVIDE INTERVALOS PARA QUE NO SUPEREN LOS 17 MINUTOS
    total_duration_s = (t_2 - t_1) / 1000  # Convertir duración total a segundos

    if total_duration_s <= 17 * 60:
        return [(t_1, t_2, clasificacion)]  # Devolver los valores originales si la duración total es menor o igual a 17 minutos

    initial_segment_duration_s = 5 * 60  # Duración inicial de los segmentos de 5 minutos en segundos

    while True:
        num_segments = int(total_duration_s // initial_segment_duration_s)
        remaining_duration_s = total_duration_s % initial_segment_duration_s  # Calcular el segmento restante
        
        if remaining_duration_s >= 0.2 * initial_segment_duration_s:
            break  # El segmento restante es mayor o igual al 20% de la duración del segmento actual

        initial_segment_duration_s -= 10  # Reducir la duración del segmento en 10 segundos

    segment_times = [
        (t_1 + i * initial_segment_duration_s * 1000, t_1 + (i + 1) * initial_segment_duration_s * 1000,clasificacion)
        for i in range(num_segments)
    ]
    # Añadir el segmento restante al final
    segment_times.append((segment_times[-1][1], t_2, clasificacion))

    return segment_times


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
def transcribir(audio_path, model , supress_s = True, previous = False, tuples = (0.0,0.2,0.4,0.6,0.8,1.0)): #RECIBE UN AUDIO Y EL MODELO WHISPER PARA TRANSCRIPCION
    print('4')
    audio_features = process_audio(audio_path)
    print('audio procesado')
    clasificacion_2 = classify_audio_segments(audio_features,modelCND2) #clasificacion que corta
    clasificacion_4 = classify_audio_segments(audio_features,modelCND4) # clasficacion que entrega información
#     clear_output(wait=True)
    intervalos2 = get_time_intervals(clasificacion_2) #tuplas [(t1,t2, clasificacion),...,]
    intervalos4 = get_time_intervals(clasificacion_4)
    temps = []
    for t1,t2,clasificacion in intervalos2:
        t1 = t1*1000
        t2 = t2*1000
        temps += dividir_segmentos(t1,t2, clasificacion)   
    intervalos = []
    for t1,t2,clasificacion in intervalos4:
        t1 = t1*1000
        t2 = t2*1000
        intervalos += dividir_segmentos(t1,t2, clasificacion)
        
    newAudio = AudioSegment.from_wav(audio_path)
    results = []

    for t1,t2, clasificacion in tqdm(temps):
        if clasificacion ==0:
            a = newAudio[t1:t2]
            a.export("temp.wav", format="wav") 
            result = model.transcribe("temp.wav", language="es", suppress_silence=supress_s, ts_num=16,condition_on_previous_text=previous)
            results.append(result.to_dict())
            print(result.to_dict())
        else:
            result = transcripcion_vacia(t1/1000,t2/1000)
            results.append(result)
    intervalos = [(t1/1000,t2/1000,clasificacion) for t1,t2,clasificacion in intervalos]
    intervalos2 = [(t1/1000,t2/1000,clasificacion) for t1,t2,clasificacion in temps]
    return results, audio_path.replace('.wav',''),intervalos2, intervalos, clasificacion_2, clasificacion_4
#     return datos, audio_path.replace('.wav','')




def transcribir_2(audio_path, model , supress_s = True, previous = False, tuples = (0.0,0.2,0.4,0.6,0.8,1.0)): #RECIBE UN AUDIO Y EL MODELO WHISPER PARA TRANSCRIPCION
    print('4')
    audio_features = process_audio(audio_path)
    print('audio procesado')
    clasificacion_2 = classify_audio_segments(audio_features,modelCND2)['clasificacion'] #clasificacion que corta
    clasificacion_4 = classify_audio_segments(audio_features,modelCND4)['clasificacion'] # clasficacion que entrega información
#     clear_output(wait=True)
    intervalos2 = get_time_intervals(clasificacion_2) #tuplas [(t1,t2, clasificacion),...,]
    intervalos4 = get_time_intervals(clasificacion_4)
    temps = []
    t2 = 0
    for l,clasificacion in enumerate(intervalos2):
        t1 = t2
        t2 = t1+10
        temps += [(t1,t2,clasificacion)]
        
    intervalos = []
    t2  = 0
    for l,clasificacion in enumerate(intervalos4):
        t1 = t2
        t2 = (t1+10)
        intervalos += dividir_segmentos(t1,t2, clasificacion)
        
    newAudio = AudioSegment.from_wav(audio_path)
    results = []

    for t1,t2, clasificacion in tqdm(temps):
        if clasificacion ==0:
            a = newAudio[t1*1000:t2*1000]
            a.export("temp.wav", format="wav") 
            result = model.transcribe("temp.wav", language="es", suppress_silence=supress_s, ts_num=16,condition_on_previous_text=previous)
            results.append(result.to_dict())
            print(result.to_dict())
        else:
            result = transcripcion_vacia(t1,t2)
            results.append(result)
    intervalos = [(t1,t2,clasificacion) for t1,t2,clasificacion in intervalos]
    intervalos2 = [(t1,t2,clasificacion) for t1,t2,clasificacion in temps]
    return results, audio_path.replace('.wav',''),intervalos2, intervalos, clasificacion_2, clasificacion_4
#     return datos, audio_path.replace('.wav','')





def json_to_dataframe(resultado):#RECIBE LISTA DE TRANSCRIPCIONES, NOMBRE DE CLASE, CLASIFICACION 4 Y ENTREGA EL DATAFRAM FINAL
    idx = resultado[1]
    results = resultado[0]
    clasificaciones = [(t1,t2,cl) for t1,t2, cl in resultado[2]]
    df = pd.DataFrame()
    last_end = 0
    k= 0
    for k, result in enumerate(results):
        segments = result['segments']
        if segments == []:
            segments = [{'text': '',
                         'start': 0,
                         'end': clasificaciones[k][1] - clasificaciones[k][0],
                         'avg_logprob': 0.0,
                         'no_speech_prob': 1.0,
                         'words': [{'word': '', 'start': 0, 'end': clasificaciones[k][1] - clasificaciones[k][0]}]}]

        data = []
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
    df['inicio'] = pd.to_datetime(df['start'], unit='s').dt.strftime('%H:%M:%S')
    df['final'] = pd.to_datetime(df['end'], unit='s').dt.strftime('%H:%M:%S')
    clasificaciones = [(t1,t2,cl) for t1,t2, cl in resultado[3]]
    clasificaciones = clasifica_tuplas(clasificaciones, list(df['start']),list(df['end']))
    A = ['ideal', 'bueno', 'regular', 'malo']
    df['clasificacion']= [A[int(i)] for t1,t2,i in clasificaciones]
    columnas =['text',
               'inicio',
               'final',
               'clasificacion',
               'start','end','avg_logprob','no_speech_prob','words','words_start','words_end','words_len']
    df = df[columnas]
    return df
def dataframe_to_xlsx(df, name):
    # Agregar una nueva columna al principio del DataFrame
    df = df.copy()
    df.insert(0, 'evaluacion', '')
    
    # Escribir el DataFrame en un archivo Excel con un writer de xlsxwriter
    writer = pd.ExcelWriter(f"{name}.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    
    # Obtener el objeto de worksheet de xlsxwriter
    worksheet = writer.sheets['Sheet1']

    # Agregar validación de datos a la columna 'evaluacion'
    n_rows = len(df)
    worksheet.data_validation('A2:A'+str(1+n_rows), {'validate': 'list', 'source': ['bueno', 'regular', 'malo']})

    # Guardar el archivo Excel
    writer.save()

    
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
