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
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        #x = self.fc1(x)
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
        self.nhid = nhid
        self.r=r
        self.tran=nn.ModuleList([Tr(self.nfeat,dropout).cuda() for i in range(self.r)])
        self.w=nn.ModuleList([Linear(self.nfeat, self.nhid) for i in range(self.r)])
        self.classifier = Mlp(self.nfeat, self.nhid*self.r, nclass,dropout)
        self.params = nn.Parameter(torch.ones(self.r*self.nfeat,self.nfeat,requires_grad=True))
    def forward(self, x):
        m=x
        x,_=self.tran[0](m)
        x=self.w[0](x)
        for i in range(1,self.r):
            y,_=self.tran[i](m)
            y=self.w[i](y)
            x=torch.cat([x,y], dim=1)
        #print(x.shape,self.params.shape)
        #x=torch.mm(x,self.params)
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits


        



