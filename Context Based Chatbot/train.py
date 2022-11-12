import json 
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

# json dosyası üzerinden veri setimizi okuyoruz.
with open('intents.json','r',encoding='utf-8') as f:
    intents = json.load(f)

all_words = [] # Tüm pattern kelimelerinin küçük harf, kökleştirme ve aynı elemanların silinerek yerleştirileceği list yapısı 
tags = [] # Tüm tag'lerin bulunacağı list yapısı
xy = [] # Train veri setini oluşturmak için pattern ve pattern'nin bulunduğu tag ifadesinin beraber bulunacağı list yapısı

# Okunan verimizi komple geziyoruz.
for intent in intents['intents']:
    tag = intent['tag']
    # print(tag)
    tags.append(tag) # Tag ifadelerini "tags" adlı değişkene atıyoruz.

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w) # Tüm pattern'lerin içerisinde bulunan kelimelerin bulunduğu list yapısı
        xy.append((w,tag)) # Pattern içerisindeki kelimelerin ait olduğu tag ile beraber tutulduğu list yapısı

ignore_words = ['?','!','.',','] # İstenmeyen ifadeler
Tr2Eng = str.maketrans("ÇĞİÖŞÜ", "çğiöşü")
# Kelimelerimize tokenizasyon işleminden sonra kökleştirme işlemi uyguluyoruz ve istenmeyen kelimeleri list yapısı içerisinden çıkartıyoruz.
all_words = [stem(w.translate(Tr2Eng)) for w in all_words if w not in ignore_words]

# Aynı geçen kelimeleri list yapısı içerisinden silerek kelimeleri harf sırasına göre sıralıyoruz.
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Eğitim veri setleri
X_train = [] # Pattern'lere ait "bag_of_words" list yapılarının tutulacağı değişken.
y_train = [] # İlgili pattern'e karşılık gelen tag ifadelerinin aynı indekste tutulacağı değişken.

# Eğitim setimizi oluşturduğumuz xy list yapısını kullanarak oluşturuyoruz.
""" 
(['Hi'], 'greeting')
X_train = [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
           0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
           0. 0. 0. 0. 0. 0.] => Bag of words içerisinde pattern yani girdiye karşılık gelen değerin 1 olduğu list yapısı
y_train = 3 => Ait olduğu tag'in tags listesindeki index karşılığı
"""

for(pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Veri setimizi DataLoader'a verebilmemiz için oluşturulan ChatDataset class yapısı.
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# Hiperparametreler
batch_size = 51
hidden_size = 51
output_size = len(tags)
input_size = len(X_train[0])
learining_rate = 0.001
num_epochs = 900

# Eğitim için hazırlanan veri setimizi eğitmek için yükleyiciye veriyoruz.
dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Model eğitiminin gerçekleştirileceği ortamı belirliyoruz.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)

# Her bir epoch(adımda) sinir ağını optimize emtke için kullancağımız yapılar
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learining_rate)

for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        # Kelime ve etiketlerimizi aygıta alıyoruz.
        words = words.to(dtype=torch.float32).to(device)
        labels = labels.to(dtype=torch.long).to(device)

        ### Forward
        # Modelimize girdilerimizi veriyoruz ve bir çıktı alıyoruz.
        outputs = model(words)
        # Modelin son halinin vermiş olduğu output değeri ile label değerini karşılaştırarak loss değerini desaplıyoruz. 
        loss = criterion(outputs, labels)

        ### Backward
        # Loss değerine bağlı olarak yapa sinir ağının kendini optimize etmesini sağlıyoruz.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if(epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

print("---------------------------------------------")
print(f'final loss = {loss.item():.4f}')
print("---------------------------------------------")

### Model dosyasının kaydedilmesi
# Eğitim işlemi sonucunda oluşan yapaty sinir ağı modelimizi ve gerekli değişkenleri kaydediyoruz.
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data,FILE)

print('Eğitim Tamamlandı. Oluşturulan model "data.pth" adıyla kaydedildi.')
print("---------------------------------------------")

