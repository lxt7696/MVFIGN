import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch.autograd import Variable
#Mlp 类是一个典型的多层感知机，包含两个全连接层 fc1 和 fc2。用于将输入特征映射到隐藏层，然后再映射到最终的类别输出。
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
        #x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_x1(x,y):
    #batch_output=F.normalize(y, p=2, dim=1)
    #batch_input=F.normalize(x, p=2, dim=1)
    aij_matrix=torch.sub(y,x).float()
    #aij_matrix=torch.sub(batch_output,batch_input).half()
    #print("?",aij_matrix)
    x1=torch.norm(aij_matrix,p='fro') 
    return x1

#Tr 类是一个简单的全连接神经网络模型，定义了一个全连接层 fc1。它在图神经网络中作为一个基本的变换模块存在。
class Tr(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(Tr, self).__init__()
        self.fc1 = Linear(hid_dim, hid_dim) 

    #forward() 方法定义了如何执行前向传播操作。通过 fc1 对输入进行线性变换，并计算节点特征之间的差异（x1）。
    def forward(self, x):
        z=x
        x = self.fc1(x)
        y=x
        x1=get_x1(z,y) #x1 的计算是通过 get_x1() 函数完成的，表示输入节点之间特征差异的 Frobenius 范数。
        return x,x1

#GMLP 是一个多层感知机（MLP）结构，包含了多个 Tr 模块，每个模块对应不同的图卷积操作。G
class GMLP(nn.Module):
    def __init__(self, r,nfeat, nhid, nclass, dropout,drate):
        super(GMLP, self).__init__()
        self.nfeat=nfeat
        self.nhid = nhid
        self.r=r
        self.tran=nn.ModuleList([Tr(self.nfeat,dropout).cuda() for i in range(self.r)])#self.tran 存储了多个 Tr 模型，用于在不同的跳数（r）上处理图的特征。
        self.w=nn.ModuleList([Linear(self.nfeat, self.nhid) for i in range(self.r)]) #self.w 存储了多个线性变换层，用于将图特征变换到隐藏层空间。
        self.classifier = Mlp(self.nhid, self.nhid, nclass,dropout) #self.classifier 是一个 MLP 模块，用于最终的分类任务。
        self.params = nn.Parameter(torch.ones(self.r,requires_grad=True))
        self.leaky_relu = nn.ModuleList([nn.LeakyReLU(drate) for i in range(self.r)])
    def forward(self, x):
        m=x
        x,_=self.tran[0](m)
        x=self.w[0](x)
        x=self.params[0]*self.leaky_relu[0](x)
        for i in range(1,self.r):
            y,_=self.tran[i](m)
            y=self.w[i](y)
            x=x+self.leaky_relu[i](y)*self.params[i]
            #x=x+y
        #x=self.leaky_relu(x)
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)
        return class_logits


        



