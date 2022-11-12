import nltk 
from nltk.stem.porter import PorterStemmer
from TurkishStemmer import TurkishStemmer
import numpy as np
# Köklendirme işlemi için gerekli olan bir metod 
stemmer = TurkishStemmer()

def tokenize(sentence): # Tokenizasyon 
    return nltk.word_tokenize(sentence)

def stem(word): # Kelime köklerine indirgeme ve harflarini küçültme
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words): # Kelime vektörünü kullanarak yapay sinir ağına verilecek veri kümesini oluşturma 
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bow   = [0 ,1 ,0 ,1 ,0 ,0 ,0]
    """
    # Gelen pattern listesime kökleştirme işlemi uyguluyoruz.
    Tr2Eng = str.maketrans("ÇĞİÖŞÜ", "çğiöşü")
    tokenized_last = [stem(w.translate(Tr2Eng)) for w in tokenized_sentence]

    # Kelime vektöre uzunluğunda çantamızı oluşturuyoruz.
    bag = np.zeros(len(all_words), dtype = np.float32)

    # Pattern listemizi gezerek çanta içerisinde kelimenin bulunduğu indeks değerlerini 1.0 yapıyoruz.
    for idx,w in enumerate(all_words):
        if w in tokenized_last:
            bag[idx] = 1.0       
    return bag






