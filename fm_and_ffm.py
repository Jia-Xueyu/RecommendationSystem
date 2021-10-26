import numpy as np
import torch.nn as nn
import torch

class FeaturesLinear(nn.Module):
    def __init__(self,field_dims,output_dim=1):
        super(FeaturesLinear,self).__init__()
        self.field_dims=field_dims
        self.output_dim=output_dim
        self.fc=nn.Embedding(sum(self.field_dims),self.output_dim)
        self.bias=nn.Parameter(torch.zeros((self.output_dim,)))
        self.offsets=np.array((0,*np.cumsum(self.field_dims)[:-1]),dtype=np.long)

    def forward(self, x):
        x=x+x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x),dim=1)+self.bias

class FeaturesEmbedding(nn.Module):
    def __init__(self,field_dims,output_dim):
        super(FeaturesEmbedding,self).__init__()
        self.embedding=nn.Embedding(sum(field_dims),output_dim)
        self.offsets=np.array((0,*np.cumsum(field_dims)[:-1]),dtype=np.long)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def foward(self,x):
        x=x+x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class FeaturizationMachine(nn.Module):
    def __init__(self):
        super(FeaturizationMachine,self).__init__()

    def forward(self, x):
        square_of_sum=torch.sum(x,dim=1)**2
        sum_of_square=torch.sum(x**2,dim=1)
        reduce_sum=torch.sum(square_of_sum-sum_of_square,dim=1,keepdim=True)*0.5
        return reduce_sum

class FM:
    def __init__(self,field_dims=None,output_dim=None):
        self.linear=FeaturesLinear(field_dims)
        self.embedding=FeaturesEmbedding(field_dims,output_dim)
        self.fm=FeaturizationMachine()

    def forward(self,x):
        x=self.linear(x)+self.fm(self.embedding(x))
        x=torch.sigmoid(x.squeeze(1))
        return x

class FieldAwareFeaturizationMachine(nn.Module):
    def __init__(self,field_dims,output_dim):
        self.field_num=len(field_dims)
        self.embedding=nn.ModuleList([nn.Embedding(sum(field_dims),output_dim) for _ in range(self.field_num)])
        self.offsets=np.array((0,*np.cumsum(field_dims)[:-1]),dtype=np.long)
        for e in self.embedding:
            nn.init.xavier_uniform_(e.weight.data)

    def forward(self,x):
        x=x+x.new_tensor(self.offsets).unsqueeze(0)
        xs=[self.embedding[i](x) for i in range(self.field_num)]
        ix=list()
        for i in range(self.field_num-1):
            for j in range(i+1,self.field_num):
                ix.append(xs[j][:,i]*xs[i][:,j])
        ix=torch.stack(ix,dim=1)
        return ix

class FFM:
    def __init__(self,field_dims,output_dim):
        self.linear=FeaturesLinear(field_dims)
        self.ffm=FieldAwareFeaturizationMachine(field_dims,output_dim)

    def forward(self,x):
        x=self.linear(x)+torch.sum(torch.sum(self.ffm(x),dim=1),dim=1,keepdim=True)
        x=torch.sigmoid(x.squeeze(1))
        return x



