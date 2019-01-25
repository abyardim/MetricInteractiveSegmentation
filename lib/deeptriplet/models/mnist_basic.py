
import torch.nn as nn
import torch.nn.functional as F

class MnistClassifier(nn.Module):
    
    def __init__(self):
        super(MnistClassifier, self).__init__()
        
        self.c1 = nn.Conv2d(1, 16, kernel_size=5)
        self.c2 = nn.Conv2d(16, 32, kernel_size=5)
        self.c3 = nn.Conv2d(32, 16, kernel_size=5)
        self.c4 = nn.Conv2d(16, 16, kernel_size=5)
        self.c5 = nn.Conv2d(16, 10, kernel_size=12)
    
    def forward(self, x):
        
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.c5(x)
        
        x = x.view(-1, 10)
        
        return x



class MnistEmbedding(nn.Module):
    
    def __init__(self, *, embed_dim):
        super(MnistEmbedding, self).__init__()
        
        self.c1 = nn.Conv2d(1, 16, kernel_size=5)
        self.c2 = nn.Conv2d(16, 32, kernel_size=5)
        self.c3 = nn.Conv2d(32, 16, kernel_size=5)
        self.c4 = nn.Conv2d(16, 16, kernel_size=5)
        self.embed_layer = nn.Conv2d(16, embed_dim, kernel_size=12)
        
        self.EMBED_DIM = embed_dim
    
    def forward(self, x):
        
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.embed_layer(x)
        
        x = x.view(-1, self.EMBED_DIM)
        
        return x
    
    def load_from_pretrained(self, cnn_pretrained):   
        params1 = cnn_pretrained.named_parameters()
        params2 = self.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)

        self.load_state_dict(dict_params2)
        

