# Proyecto de transcripción de audio
Este proyecto consiste en un programa que permite transcribir audios y dar un output en un formato de dataframe determinado.

## Instalación
Para utilizar este programa, se deben instalar los paquetes necesarios desde el archivo requirements.txt. Para instalar los paquetes, ejecuta el siguiente comando en la terminal:
```python
pip install -r requirements.txt
````
Además se debe instalar __Stable Whisper__ 

```python
pip install -U stable-ts
```
para la última versión
```python
pip install -U git+https://github.com/jianfch/stable-ts.git
```
Más información en [La página de Stable Whisper](https://github.com/jianfch/stable-ts#setup)
## Uso
Para utilizar el programa solo para transcribir, sigue los siguientes pasos:

1. Importa los módulos necesarios de la carpeta Transcriptor del archivo Transcriptor.py:
  ```python
  from Transcriptor import *
  ````
2. A partir del archivo mp4 puedes crear el archivo wav con la función 
  ```python
create_wav("\video_path.mp4","\audio_path.wav")
  ```
2. Define la variable __'modelo'__ según alguno de los modelos de Whisper y carga el modelo con la función __run_model__
  <div style="text-align:center"><img src="./whisper-models.png" alt="Imagen de ejemplo" style="max-width:500px; height:200px;"></div>

  ```python
  modelo = "base"
  model = run_model(modelo)
  ````
3. Ejecuta la función __transcribir__ con los parámetros definidos anteriormente:

  ```python
  result_json, idx = transcribir(audio_path, model)
  ````
4. Finalmente con la función __json_to_dataframe__ crea el dataframe que resume la información del json:
  ```python
  dataframe = json_to_dataframe(result_json, idx) 
  ````
5. Guarda el dataframe
  ```python
  dataframe.to_csv(idx.csv)
  ````
