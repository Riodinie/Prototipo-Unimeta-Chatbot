# Implementación de Chatbot con Flask y JavaScript


Esto ofrece 2 opciones de implementación:
- Implementar dentro de la aplicación Flask con la plantilla jinja2
- Sirve solo la API de predicción de Flask. Los archivos html y javascript usados ​​se pueden incluir en cualquier aplicación Frontend (con solo una ligera modificación) y luego pueden ejecutarse completamente separados de la aplicación Flask.

## Configuración inicial:
Este repositorio contiene actualmente los archivos de inicio.

Clonar repositorio y crear un entorno virtual.
```
python3 -m venv venv
.\venv\Scripts\activate
```
Instalar dependencias
```
(venv) pip install nltk
(venv) pip install Flask torch torchvision nltk
```
Instalar el paquete nltk
```
(venv) python
>>> import nltk
>>> nltk.download('punkt')
```
Modifica `intents.json` con diferentes `intents`  y respuestas para tu Chatbot.

Ejecutar
```
python train.py
```
Esto volcará el archivo `data.pth`. Y luego ejecute el siguiente comando para probarlo en la consola.
```
python chat.py
```


## Créditos:
Este repositorio se usó para el código frontend:
1. https://github.com/hitchcliff/front-end-chatjs
2. https://templatemo.com/tm-569-edu-meeting#goog_rewarded
