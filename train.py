# Librerias
import random                                       # Respuestas Aleatorias 
import json                                         # Archivos JSON
import torch                                        # Deep Learning
import torch.nn as nn                               # Modulo RNA
import numpy as np                                  # Calculos matematicos y @

from torch.utils.data import Dataset, DataLoader    # Manejo de conjunto de datos y cargarlos
from nltk_utils import bag_of_words, tokenize, stem # Funciones de procesamiento de Texto (NLP/Palabras a Vector/Mensaje a Palabras/descomponer palabras)
from model import NeuralNet                         # Importa Modelo RNA Personalizada

# Leyendo el Json (Intenciones)
with open('intents.json', 'r') as f:
    intents = json.load(f)              # JSON a un diccionario

# Inicializacion de Listas
all_words = []                          # Lista Palabras Unicas
tags = []                               # Lista etiquetas de Intencion
xy = []                                 # Lista pares de Patrones (frases)

# Iterar a travez de cada patron de Intencion (JSON)
for intent in intents['intents']:
    tag = intent['tag']                 # Obtiene la etiqueta de intención
    tags.append(tag)                    # Agrega la etiqueta a la lista de etiquetas

    for pattern in intent['patterns']:  # Itera a través de los patrones de conversación
        w = tokenize(pattern)           # Tokeniza cada patrón en palabras
        all_words.extend(w)             # Agrega las palabras a la lista de todas las palabras unicas
        xy.append((w, tag))             # Agrega el par (palabras, etiqueta) a la lista de patrones

# Aplica stemming y convierte las palabras a minúsculas
ignore_words = ['?', '.', '!']          # Palabras que se ignorarán
all_words = [stem(w) for w in all_words if w not in ignore_words]   # Reduce las palabras a su raíz

# Convierte la lista en un conjunto para eliminar duplicados y ordena
all_words = sorted(set(all_words))      # para Palabras Unicas
tags = sorted(set(tags))                # para Etiquetas

# Imprimir Resultados
print(len(xy), "patterns")              # Cuántos patrones encontrados
print(len(tags), "tags:", tags)         # Cuántas etiquetas únicas hay
print(len(all_words), "unique stemmed words:", all_words)   # Cuántas palabras únicas hay

# Creación de Datos de Entrenamiento
X_train = []                            # Lista para las características (X)
y_train = []                            # Lista para las etiquetas (y)

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
