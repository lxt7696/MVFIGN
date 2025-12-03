import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch.autograd import Variable
class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim,c_dim,dropout):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim,c_dim)
        #self.fc2 = Linear(input_dim,c_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)
        #self.layernorm = LayerNorm(c_dim, eps=1e-6)
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_x1(x,y):
    #batch_output=F.normalize(y, p=2, dim=1)
    #batch_input=F.normalize(x, p=2, dim=1)
    aij_matrix=torch.sub(y,x).half()
    #aij_matrix=torch.sub(batch_output,batch_input).half()
    x1=torch.norm(aij_matrix,p='fro') 
    return x1

class Tr(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(Tr, self).__init__()
        self.fc1 = Linear(hid_dim, hid_dim) 

    def forward(self, x):
        z=x
        x = self.fc1(x)
        y=x
        x1=get_x1(z,y)
        return x,x1

class GMLP(nn.Module):
    def __init__(self, r,nfeat, nhid, nclass, dropout,drate):
        super(GMLP, self).__init__()
        self.nfeat=nfeat
        self.sum_l=sum(nfeat)
        self.nhid = nhid
        self.r=r
        # nfeat是一个列表，每个客户端特征维度不同
        self.tran=nn.ModuleList([Tr(self.nfeat[i],dropout).cuda() for i in range(self.r)])
        self.classifier = Mlp(self.sum_l, self.nhid, nclass,dropout)
        self.leaky_relu = nn.ModuleList([nn.LeakyReLU(drate) for i in range(self.r)])
    def forward(self, x):
        m=x
        # x是多个客户端特征的列表
        x,_=self.tran[0](m[0])
        x=self.leaky_relu[0](x)
        for i in range(1,self.r):
            y,_=self.tran[i](m[i])
            x=torch.cat([x,self.leaky_relu[i](y)], dim=1) # 特征拼接 #直接拼接，没有MPC?
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits


        



