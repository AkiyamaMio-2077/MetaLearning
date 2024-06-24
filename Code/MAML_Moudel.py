
import torch.nn.modules as nn
import torch 

def maml_init_(module):
    torch.nn.init.xavier_uniform(module.weight.data,gain = 1.0) # 对xavier进行初始化，使其变成均匀分布，U (-a,a), a = gain * 6fan_in + fan_out
    torch.nn.init.constant_(module.bias.data , 0.0)             # 用将bias的data 这个数组填充为0    

    return module


class ConvBlock (nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,max_pool_factor = 1.0):
        super().__init__()
        stride = (int(2*max_pool_factor))   
        self.max_pool = nn.MaxPool1d(kernel_size= stride, stride= stride ,ceil_mode= False)  #卷积核大小为 1* stride， 模式为地板模式
        self.normalize = nn.BatchNorm1d(out_channels,affine  = True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        
        self.conv = nn.Conv1d(in_channels, out_channels , kernel_size, stride = 1 ,padding = 1,bias= True )
        maml_init_(self.conv)

    def forward(self,x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):
    def __init__(self,hidden = 64 ,layers = 4,channels = 1 ,max_pool_factor = 1.0):
        
        core = [ConvBlock(channels,hidden,3,max_pool_factor)]
        for _ in range (layers - 1):
            core.append((ConvBlock(hidden,hidden,3,max_pool_factor)))  #这里有问题 
        super (ConvBase,self).__init__(*core)


class CNN4Backbone(ConvBase):
    def forward(self, x):
        x = super(CNN4Backbone,self).forward(x)
        x = x.reshape(x.size(0),-1)
        return x



class Net4CNN(torch.nn.Module):
    def __init__(self , output_size,hidden_size,layers,channels,embedding_size):
        print(output_size,hidden_size,layers,channels,embedding_size)
        super().__init__()
        self.features = CNN4Backbone(hidden_size,layers,channels,4//layers) 
        self.classifier = torch.nn.Linear (embedding_size,output_size,bias = True)
        maml_init_(self.classifier)
        self.hidden_size = hidden_size
    
    def forward(self,x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.features(x)
        feat = x
        x = self.classifier(x)
        return x

# model = Net4CNN(output_size= 4, hidden_size = 64,layers= 4, channels= 1 ,embedding_size = 1024)
# print(model)