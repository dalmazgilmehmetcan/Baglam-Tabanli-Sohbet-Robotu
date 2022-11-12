import random
import json
import torch 
from model import NeuralNet
from nltk_utils import *

# Model çalışırken hangi ortamda çalışacağını belirliyoruz. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Veri setimizi okuyoruz.
with open('intents.json','r',encoding='utf-8') as f:
    intents = json.load(f)

# Eğitilmiş modelimizi alıyoruz.
FILE = "data.pth"
data = torch.load(FILE)

# Model dosyası içerisinde bulunan değerleri değişkenlere kaydediyoruz.
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Eğitilmiş modelimizi belirtilen aygıt üzerinde ayağa kaldırıyoruz.
model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Chatbot"
print("Merhaba. Nasıl yardımcı olabilirim ? (Çıkış için 'q')")
print("--------------------------")

Tr2Eng = str.maketrans("ÇĞİÖŞÜ", "çğiöşü")

while True:
    sentence = input("Siz: ")
    if sentence == "q":
        break
        
    print(f"Girdi => {sentence}")
    print("--------------------------")
    # 1) Kullanıcı tarafından girilen cumleye tokenizasyon ediliyor.
    sentence = tokenize(sentence)
    print(f"Tokenizasyon gerçekleştirilmiş hali => {sentence}")
    print("--------------------------")

    # 2) Tokenize edilmiş kelimelerin bag of word'u oluşturuluyor. Not: bag_of_words metodu içerisine bakınız.
    X = bag_of_words(sentence,all_words)
    
    tokenized_sentence = [stem(w.translate(Tr2Eng)) for w in sentence]
    print(f"Girdiye ait veri ön işleme sonucu(tokenize + stem + lower) => {tokenized_sentence}")
    
    
    tutucu = []
    for i in range(len(X)):
        if X[i] == 1:
            tutucu.append(all_words[i])
    print(f"Bag of words içerisinde eşleşen kelimeler => {tutucu}")
    print("--------------------------")
    # print(f"Bag of Words gerçekleştirilmiş hali => {X}")
    # print("--------------------------")

    # 3) Oluşan list bir numpy list. Listeyi torch tipine çevirebilmek için reshape uygulanıyor. 
    X = X.reshape(1,X.shape[0])

    # 4) Reshape uygulanan liste torch tipine çevriliyor.
    X = torch.from_numpy(X)

    # 5) Modele verilmeye hazır olan liste yapısı modele girdi olarak veriliyor. Model bize bir çıktı ifadesi üretecek.
    output = model(X)

    # 6) Modelin üretmiş olduğu tahmin değeri alınıyor. Bu değer bir index değeri olup tag listesi içindeki bir indexi ifade etmektedir.
    _,predicted = torch.max(output, dim = 1)

    # 7) Predicted içerisinde bulunan index değerine karşılık gelen tag ifadesi alınıyor. Bu tag ifadesi modelimizin girdiye bağlı olarak ürettiği tahmindir. 
    # Bu değerin karşılığında olan response değerlerden biri rastgele olarak kullanıcıya döndürülmektedir.
    tag = tags[predicted.item()]
    print(f"Modelin belirlediği başlık sonucu => {tag}")
    print("--------------------------")

    # 8) Modelimiz tarafıdan gerçekleştirilen tahmine ait oluşan softmax değeri alınıyor.
    # Bu değer 0-1 arasında yer alan ve modelimizin yaptığı tahmine ait kararlılığını gösteren bir değerdir. 
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]
    print(f"Belirlenen başlığa ait tahmin değeri => {prob}")
    print("--------------------------")
    
    # Model tahmin değeri(prob) 0.75'ten büyükse bulunan tag değerine ait response değeri kullanıcıya gösterilmektedir. 
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Ne dediğinizi anlayamadım")
