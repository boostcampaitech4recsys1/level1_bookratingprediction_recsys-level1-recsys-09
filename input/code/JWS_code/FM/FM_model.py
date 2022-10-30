import numpy as np

import torch
import torch.nn as nn

def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


class FactorizationMachine(nn.Module):
# FM layer
    def __init__(self, reduce_sum:bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
    
class _FactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x: torch.Tensor):
        x = self.linear(x) + self.fm(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)
    
class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        try:
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return self.embedding(x)
        except:
            x = np.delete(x.to('cpu'),2,axis=1)
            x = torch.tensor(x).to('cuda')
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return self.embedding(x)

class FeaturesLinear(nn.Module):

    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x: torch.Tensor):
        try:
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return torch.sum(self.fc(x), dim=1) + self.bias
        except:
            x = np.delete(x.to('cpu'),2,axis=1)
            x = torch.tensor(x).to('cuda')
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return torch.sum(self.fc(x), dim=1) + self.bias