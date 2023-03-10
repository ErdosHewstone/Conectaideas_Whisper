from pyannote.audio import Pipeline  # importar la clase Pipeline de la librería pyannote.audio

# Definir función para cargar el modelo pre-entrenado de speaker diarization
def run_pipeline(token='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'):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)
    # Se crea una instancia de la clase Pipeline a partir del modelo pre-entrenado de la librería pyannote.audio
    # La variable token se usa para autenticar el acceso al modelo pre-entrenado en Hugging Face
    return pipeline

# Definir función para crear un dataframe a partir de un diccionario de hablantes y tiempos
def create_dataframe(data, string_id):
    dfs = []  # Se crea una lista vacía para almacenar dataframes
    for hablante, tiempos in data.items():  # Se itera sobre el diccionario
        rows = [[string_id, hablante, tiempo[0], tiempo[1]] for tiempo in tiempos]  # Se crea una lista de filas para cada hablante
        df = pd.DataFrame(rows, columns=["id", "Hablante", "Inicio", "Final"])  # Se crea un dataframe a partir de la lista de filas
        dfs.append(df)  # Se agrega el dataframe a la lista de dataframes
    return pd.concat(dfs, ignore_index=True)  # Se concatenan todos los dataframes de la lista en uno solo

# Definir función para identificar los hablantes en un archivo de audio
def speakers(audio_path):
    pipeline = run_pipeline()  # Se carga el modelo pre-entrenado de speaker diarization
    diarization = pipeline(audio_path)  # Se aplica el modelo al archivo de audio para identificar los hablantes
    Speakers = {}  # Se crea un diccionario vacío para almacenar los tiempos de cada hablante
    for turn, _, speaker in diarization.itertracks(yield_label=True):  # Se itera sobre las pistas de audio del resultado de la diarización
        if speaker in Speakers:  # Si el hablante ya está en el diccionario, se agrega el tiempo a la lista correspondiente
            Speakers[speaker].append([turn.start, turn.end])
        else:  # Si el hablante no está en el diccionario, se crea una nueva entrada con el tiempo
            Speakers[speaker] = [[turn.start, turn.end]]
    return create_dataframe(Speakers, string_id=audio_path)  # Se crea un dataframe a partir del diccionario de hablantes y tiempos, y se retorna

 
  
