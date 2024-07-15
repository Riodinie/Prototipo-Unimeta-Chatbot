# Implementación de Chatbot con Flask y JavaScript

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
##Imganes 
![Captura de pantalla_12-7-2024_13171_127 0 0 1](https://github.com/user-attachments/assets/0ca5bedf-a045-418f-b6a2-a5b28a1f7d49)
![Captura de pantalla_12-7-2024_131954_127 0 0 1](https://github.com/user-attachments/assets/9ee7be89-c853-435f-86fb-f0ded18a95a0)
## Créditos:
Este repositorio se usó para el código frontend:
1. https://github.com/hitchcliff/front-end-chatjs
2. https://templatemo.com/tm-569-edu-meeting#goog_rewarded
