# Proyecto de transcripción de audio
Este proyecto consiste en un programa que permite transcribir audios y dar un output en un formato de dataframe determinado.

## Instalación
Para utilizar este programa, se deben instalar los paquetes necesarios desde el archivo requirements.txt. Para instalar los paquetes, ejecuta el siguiente comando en la terminal:
```python
pip install -r requirements.txt
````
## Uso
### Solo para transcribir
Para utilizar el programa solo para transcribir, sigue los siguientes pasos:

1. Importa los módulos necesarios de la carpeta Transcriptor del archivo Transcriptor.py:
  ```python
  from Transcriptor import run_model, Trans_Alpha
  ````
2. Define la variable __'modelo'__ según alguno de los modelos de Whisper
  <div style="text-align:center"><img src="./whisper-models.png" alt="Imagen de ejemplo" style="max-width:500px; height:200px;"></div>

  ```python
  modelo = "base"
  ````
3. Ejecuta la función __Trans_Alpha__ con los parámetros definidos anteriormente:

  ```python
  DataFrame = Trans_Alpha(tiempo, Inicio, Final, audio_path, audio_id, model)
  ````

### Diarización y transcripción

Para utilizar el programa para diarización y transcripción, sigue los siguientes pasos:

1. Importa los módulos necesarios de las carpetas __Transcriptor__ y __Diarization__:
  ```python
from Transcriptor import run_model, data_transcriptor
from Diarization import speakers

  ````
  
2. Define el siguiente parámetro:
    * __audio_path__: La ruta del archivo de audio que deseas transcribir.
3. Ejecuta la función __speakers__ con el parámetro __audio_path__ para obtener un DataFrame que contiene las diarizaciones:
  ```python
df = speakers(audio_path)

  ````
4. Define la variable __'modelo'__ según alguno de los modelos de Whisper
  <div style="text-align:center"><img src="./whisper-models.png" alt="Imagen de ejemplo" style="max-width:500px; height:200px;"></div>

  ```python
  modelo = "base"
  ````
  
5. Ejecuta la función __data_transcriptor__ con los siguientes parámetros:
  * __df__: El DataFrame que contiene las diarizaciones.
  * __model__: El modelo de lenguaje utilizado para la transcripción.
  * __audio_path__: La ruta del archivo de audio que deseas transcribir.
 
  ```python
  DataFrame = data_transcriptor(df, model, audio_path)
  ````
