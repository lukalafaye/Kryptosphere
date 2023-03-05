# 1. PyTorch Tensors and Datasets

Equivalent PyTorch des arrays Numpy. Les tensors ne contiennent que des éléments du même type.

## Les bases

### Les types de tensors
```
import torch
a = torch.tensor([0.0,1.0,2.0,3.0]) # Crée un tensor à partir d'une liste python
a = torch.tensor([0.0,1.0,2.0,3.0], dtype=torch.int32) # Crée un tensor à partir d'une liste python après le cast des élements à dtype

type(tensor) # Types de tensor != type des éléments stockés
```

- FloatTensor pour les floats 32-bit 
- Double tensor pour les floats 64-bit 
- HalfTensor pour les floats 16-bit 
- ByteTensor pour les uint8

### Taille et dimension
```
a = torch.FloatTensor([0,1,2,3])
a.type() -> torch.FloatTensor
a -> tensor([0.,1.,2.,3.])

a.size() -> 4; nombre d'éléments dans le tensor
a.ndimension() -> 1; number of dimensions
```

### Reshape un tensor

```
a = torch.FloatTensor([0,1,2,3])
a_col = a.view(4, 1) # .view(nlines, ncols) ; If nlines=-1 adds all lines
```

Attention à bien sauvegarder le résultat dans un nouveau tensor.

### Conversions avec Numpy

```
numpy_array = np.array([0.0,1.0,2.0,3.0]) # Modifier numpy_array => modifier torch_tensor => modifier back_to_numpy
torch_tensor = torch.from_numpy(numpy_array) # torch_tensor pointe vers numpy_array
back_to_numpy = torch_tensor.numpy() # back_to_numpy pointe vers torch_tensor
```

Remarque : En numpy, pour mettre tous les éléments d'un array à 0, on peut utiliser `numpy_array[:] = 0`.

### Conversion avec Pandas

```
pandas_series = pd.Series([0.1,2,0.3,10.1]) # Les objets peuvent être de n'importe quel type
pandas_to_torch = torch.from_numpy(pandas_series.values)
```

### Conversion en liste 

```
this_tensor = torch.tensor([0,1,2,3])
torch_to_list = this_tensor.tolist() # torch_to_list: [0,1,2,3]
```

### Conversion des éléments en nombres

Les éléments d'un tensor sont aussi des tensors : 

```
new_tensor = torch.tensor([5,2,6,1])
new_tensor[0] -> tensor(5)
new_tensor[1] -> tensor(2)
```

Pour convertir un tensor de dimension 1 en nombre, on peut utiliser `new_tensor[0].item()` -> 5.

### Slicing et indexage

Fonctionne comme les listes Python. On peut changer un ensemble de valeurs en utilisant le slicing :

```
c = torch.tensor([100,1,2,3,0])
d = c[1:4] # d = torch.tensor([1,2,3])
c[3:5] = torch.tensor([300.0,400.0]) # c -> tensor([100,1,2,300,400])
```

Pour sélectionner des indices précis :

```
selected_indexes = [1, 3]
c[selected_indexes] = 1000 -> tensor([100,1000,2,1000,0])
```

### Linspace

Pour générer un tensor de 9 nombres équitablement répartis entre -2 et 2 inclus : `torch.linspace(-2,2,steps=9)`.

### Opérations en 1D

- Additionner des tensors de même type -> `u+v`
- Multiplier par un scalaire -> `2*u`
- Faire un produit des éléménts de même indice -> `u*v`, 
- Faire un produit scalaire (dot product) -> `torch.dot(u,v)`
- Ajouter une constante au tensor -> `u+1`
- Faire la moyenne des éléments -> `u.mean()`
- Trouver le max -> `u.max()`
- Appliquer des fonctions -> `torch.sin(u)`

### Opérations en 2D

Soit un A un torseur de dimension 2.
`A = [[11,12,13], [21,22,23], [31,32,33]]` par exemple.

- `A.ndimension()` -> 2
- `A.shape` -> `torch.Size(3,3)`
- `A.size` -> (3,3)
- `A.numel` -> 9 ; nombre total d'éléments

Soient deux tensors `u `et `v` de dimension 2.

- Additionner des tensors de même type -> `u+v`
- Multiplier par un scalaire -> `2*u`
- Faire un produit des éléménts de même indice -> `u*v`, 
- Faire un produit scalaire (dot product) -> `torch.dot(u,v)`
- Multiplier u et v -> w = `torch.mm(u,v)`

Indexage en 2D : 
`print(A[1, 2])` est équivalent à `A[1][2]`.

### Calcul de dérivées

```
x = torch.tensor(2, requires_grad=True)
y = x**2
y.backward -> Calcule la dérivée de x en tant que variable de la fonction y, en réalité il s'agit de la différentielle dy/dx Affecte cette valeur dans la variable x
x.grad -> calcule en x=2
```

### Forward and backword propagation

```
class SQ(torch.autograd.Function):
 	@staticmethod
    def forward(ctx,i):
        """
        ctx est un objet de contexte utilisé pour garder en cache des informations utilisées lors de la back propagation que l'on récupère grâce à ctx.save_for_backward method.
        """
        result=i**2
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_output = 2*i
        return grad_output

x=torch.tensor(2.0,requires_grad=True)
sq=SQ.apply # Instancie la classe SQ 
y=sq(x) # Forward propagation

y.backward() # Backward propagation
x.grad # Affichage de la dérivée évaluée en x
``` 

### Créer et transformer un dataset

```
from torch.utils.data import Dataset

class toy_set(Dataset): # Sous classe de Dataset
  def __init__(self, length=100, transform=None):
    self.x = 2*torch.ones(length,2) # Génère un tensor contenant length lignes [1,1]*2
    self.y = torch.ones(length,1)
    self.length = length
    self.transform = transform
  def __getitem__(self, index):
    sample = self.x[index], self.y[index]
    if self.transform: # if None = False
      sample = self.transform(sample)
    return sample
  def __len__(self):
    return self.len
```

Créons une transformation qui prend un sample (x, y) et effectue des opérations sur ces deux variables :

```
class add_mult(object):
  def __init__(self, addx=1, muly=1):
    self.addx = addx
    self.muly = muly
  def __call__(self, sample):
    x=sample[0]
    y=sample[1]
    x=x+self.addx
    y=y*self.muly
    sample=x,y
    return sample
```

Appliquons notre transformation :

```
dataset=toy_set()
a_m=add_mult() # instance de classe

x_,y_ = a_m(dataset[0]) # x_ notation pour x transformé
([2,2],1) -> ([3,3],1)


a_m = add_mult()
Pour remplacer tous les éléments de dataset : dataset_ = toy_set(transform=a_m)
```

### Composer des transformations

```
from torchvision import transforms 
data_transform = transforms.Compose([constructeur1, constructeur2, ...]) # exemple de constructeur : add_mult() -> __call__()
```

### Créer des datasets d'images 

Librairies utilisées :
```
from PIl import Image # Image.open(path)
import pandas as pd
import os
from matplotlib.pyplot import imshow
from torch.utils.data import Dataset, Dataloader
```
		
Astuce : 
Créer un chemin à partir de string "Dir/" ou "Dir" avec `os.path.join(chem1, chem2, chem3)`

Lire le csv dans un dataframe pandas : 

```
data_name = pd.read_csv(csv_path)
data_name.head() # Premières lignes du dataframe
```

Extraire une valeur d'indice i,j d'un dataframe avec `data_name.iloc[i,j]`.


### Afficher une image

```
image = Image.open(image_path) # input: chemin, output: objet image
plt.imshow(image,cmap="gray", vmin=0, vmax=255)
plt.title(titre_str)
plt.show()
```

cmap : map data to scalars
vmin,vmax définissent le data range

### Transformations sur les images

```
import torchvision.transforms as transforms
transforms.CenterCrop(20) -> crop l'image au centre afin d'obtenir 20x20
transforms.ToTensor() -> convertir l'image en tensor 3D de valeurs comprises entre 0 et 1 (RGB / 255.0)
transforms.RandomVerticalFlip(p=1) p est une proba... ici flip l'image upside down
```

On peut maintenant composer ces transformations :

```
croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
```

### Datasets préfabriqués pour comparer des modèles

La base de données MNIST est une grande base de données de chiffres manuscrits qui est couramment utilisée pour la formation de divers systèmes de traitement d'images : 

```
import torchvision.datasets as dsets

dataset = dsets.MNIST(root="./data", train=False, download=True, transform = transforms.ToTensor())
# MNIST dataset = train/teste, télécharger/déjà téléchargé
```

