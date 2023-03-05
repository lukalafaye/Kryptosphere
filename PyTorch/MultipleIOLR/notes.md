# Régression linéaire avec plusieurs I/O

## Rappel : faire plusieurs prédictions

```
import torch.nn as nn

class LR(nn.Module):
  def __init__(self, input_size, output_size):
    super(LR, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
 
 	def forward(self, x):
 	  out = self.linear(x)
 	  return out
```

`model = LR(2, 1)`est équivalent à : 

```
from torch.nn import Linear
model = Linear(input_size=2, output_size=1)
```

On peut alors appliquer le modèle pour obtenir plusieurs prédictions : 

```
X = torch.tensor([[1,2],[3,4]])
Yhat = model(X)
-> tensor([[0.2],[0.3]])
```

`model.state_dict()` affiche les paramètres du modèle (weights W et bias b)

## Entrainement d'un modèle de régression linéaire possédant plusieurs outputs

On crée un Dataset adapté : 

```
from torch.utils.data import Dataset, DataLoader
class Data2D(Dataset):
	def __init__(self):
		self.x = torch.zeros(20,2)
		self.x[:,0] = torch.arange(-1,1,0.1)
		self.x[:,1] = 