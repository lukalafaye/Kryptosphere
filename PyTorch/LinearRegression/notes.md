# Linear regression

## Définitions

Il existe trois catégories de descente de gradient : 
- Batch Gradient Descent : 1 itération = 1 epoch = on parcout tout les samples du Dataset de manière linéaire, il n'y a qu'un seul batch constitué du Dataset entier
- Stochastic Gradient Descent : 1 itération = 1 sample séléctionné de manière aléatoire (pratique pour les gros jeux de données)
- Mini-batch Gradient Descent : 1 mini batch est composé de plusieurs samples, 1 itération = utilisation d'un mini batch = plusieurs samples (les opérations sont vectorizées)

Rq : 1 itération = forward + backward propagation

## Linear

```
from torch.nn import linear

model = Linear(in_features=1, out_features=1)
print(list(model.parameters())) # -> [slope w, bias b]

x = torch.tensor([0.0])
yhat = model(x)
```

On peut appeler ce modèle sur un tensor de plusieurs lignes et chaque ligne sera MaJ.

## Custom modules 

Il s'agit de classes, enfants du module nn. Créons un custom module LR : 

```
class LR(nn.Module):
  def __init__(self, in_size, output_size):
    super(LR, self).__init__() # Permet de créer des champs défaut de nn.Module sans les expliciter
    self.linear = nn.Linear(in_size, output_size) # Définit notre modèle linéaire
  def forward(self, x): # Joue le même rôle que call. x est un tuple contenant in_size et out_size
    out = self.linear(x)
    return out
```

Créons un modèle à partir de notre custom module et initialisons w et b :

```
model = LR(1,1)
model.state_dict()["linear.weight"].data[0] = torch.tensor([0.5]) 
model.state_dict()["linear.bias"].data[0] = torch.tensor([-1])

model.state_dict() est une méthode importante qui retourne un dictionnaire contenant les paramètres de model
model.state_dict().keys()
model.state_dict().values()
```

Affichons les paramètres :

```
print(list(model.parameters()))
```

## Linear regression training

Définiton d'un epoch = on parcourt tout le jeu de données une fois.
Ici, 1 it sur un batch constitue un epoch et permet de mettère à jour les w_i.

Fonction de loss : $l(w,b)=\frac{1}{N} \times \sum(yn - w xn - b)^2$


```
def criterion(yhat, y): # Fonction de loss 
  return torch.mean((yhat-y)**2)
  
for epoch in range(15):
  Yhat=forward(X)
  loss=criterion(Yhat,Y)
  loss.backward()
  
  w.data = w.data - lr*w.grad.data
  w.grad.data.zero_()
  
  b.data = b.data - lr*b.grad.data
  b.grad.data.zero_()
```
  
## Stochastic gradient descent and data loader

-> Au lieu de calculer le vrai gradient en prenant tout les samples d'un coup, on le calcule sur chaque sample un par un

```
dataset=Data()
trainloader=DataLoader(dataset=dataset, batch_size=1)
```

On peut maintenant écrire :

```
for x,y in trainloader:
  yhat = forward(x)
  loss = criterion(yhat, y)
  loss.backward()
```

Remarque :
```
torch.meshgrid(a,b) # a et b deux tensors (N,1)
-> renvoie un tuple contenant deux tensors
-> tensor1 = [a,a,..,a] N fois
-> idem pour 2
```

## Optimisation

```
from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.01) # Met à jour le lr
optimizer.state_dict() # Affiche les paramètres du modèle

for epoch in range(100):
  for x,y in trainloader:
    yhat = model(x)
    loss = criterion(yhat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # met à jour les params + les weights
```

## Entrainement, validation, test split

Overfitting : le modèle prédit bien un ensemble limité de points mais n'est pas du tout accurate en dehors de cet ensemble.

Signe de overfitting : le modèle performe trop bien sur le jeu d'entrainement et pas bien sur d'autres jeux (test/validation).

Pour éviter d'overfit, on split le dataset en training, testing, validation.
On train avec différents lr et pour chaque lr on teste sur le validation dataset pour regarder quel lr minimise le cost et si cela correspond au même lr que pour le training set.

Paramètres du modèles : les w_i, b_i
Hyperparameters : BS mini batch size, etha learning rate

Fonction argmin : 
w*, b* = argmin { l(w,b) }
argmin retourne le minimum des valeurs inputs possibles pour obtenir 
X* veut dire qu'on max.

Ce code permet de comparer test et validation pour différents lr : 
```
epochs=100
learning_rates = [0.00001,0.0001, ...]

validation_error = torch.zeros(len(learning_rates)
test_error = torch.zeros(len(learning_rates)

MODELS = []

from torch import optim
for i, learning_rate in enumerate(learning_rates):
  model = LR(1,1)
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  for epoch in range(epochs):
            for x, y in trainloader:
                yhat = model(x)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

  yhat = model(train_data.x)
  loss = criterion(yhat, train_data.y)
  test_error[i] = loss.item()

  yhat = model(val_data.x)
  loss = criterion(yhat, val_data.y)
  validation_error[i] = loss.item()

  MODELS.append(model)
```

### Mini-batch Gradient Descent

Pareil que le stochastic gradient descent mais on modifie le batch_size à qqc plus grand que 1...