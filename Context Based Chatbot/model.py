import torch 
import torch.nn as nn

class NeuralNet(nn.Module):
    # Yapay sinir ağı modelimizi tanımlıyoruz. 
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # 2 ara katmana sahip bir sinir ağı yapısı
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU() # Aktivasyon fonksiyonumuz
    
    def forward(self, x):
        # Sinir ağımıza input layer'dan pattern'a göre oluşturulan bag_of_word list yapımızı veriyoruz.
        out = self.l1(x)
        # Bir sonraki layer'a ilerleme gerçekleşmeden önce nöron içerisinde bulunan aktivasyon fonksiyonuna output değeri verilmektedir.
        # Aktivasyon fonksiyonunun vermiş olduğu çıktı değeri bir sonraki layer'ın girdisi olmaktadır.
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # Son çıkış değeri için aktivasyon fonksiyonuna veya softmax'a ihtiyaç yoktur. Sinir ağı son durumu kendi ayarlamaktadır.
        return out


