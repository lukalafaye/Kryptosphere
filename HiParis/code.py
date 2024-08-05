# Convert to ipynb to use

#!/usr/bin/env python
# coding: utf-8

# In[9]:


### On importe les différentes librairies
# import cudf
from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from collections import deque
import random
import time
pd.set_option('display.max_columns', None)
import random as rd
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from collections import deque
import random
import time
import datetime
pd.set_option('display.max_columns', None)
get_ipython().system('pip install xgboost')


# In[10]:


# On ouvre le jeu de données
datas = pd.read_csv('./train_features_sent.csv')
labels = pd.read_csv('./train_labels_sent.csv')


# In[11]:


datas["energy_consumption_per_annum"] = labels["energy_consumption_per_annum"] # Ajoute energy à la fin de datas


# In[12]:


labels.head() # Output


# In[13]:


datas.head(1) # Input


# In[109]:


datas.replace('', np.nan, inplace=True)
datas.replace([],np.nan,inplace=True)
datas.replace('[]',np.nan,inplace=True)
datas.replace('[NULL]',np.nan,inplace=True)
#datas.fillna("nan", inplace = True)


# In[15]:


datas.head()


# In[16]:


print(labels.shape)
print(datas.shape)


# In[160]:


# Helper

def is_categorical(array_like): # Text or mixed numeric and non-numeric values
    return array_like.dtype.name == 'object'

def is_numerical(array_like):
    return (array_like.dtype.name == 'int64' or array_like.dtype.name == 'float64')

def is_bool(array_like):
    return(array_like.dtype.name == 'bool')


# In[204]:


i=-1
def change_cell(cell):
    global i
    i+=1
    return (cell,i)

def cleandata(a, b):
    datas = pd.read_csv('./train_features_sent.csv')
    labels = pd.read_csv('./train_labels_sent.csv')
    datas.replace('', np.nan, inplace=True)
    datas.replace([],np.nan,inplace=True)
    datas.replace('[]',np.nan,inplace=True)
    datas.replace('[NULL]',np.nan,inplace=True)
    clean_data = datas.drop(["building_period", "building_use_type_code", "is_crossing_building", "area_code", "post_code", "level_0"], axis=1)
    clean_data = clean_data.iloc[a:b]
    
  # N = 25 # seuil d'instabilite
  # for index, row in clean_data.iterrows():
  #     if row.isna().sum() >= N:
  #       #print(index)
  #       clean_data.drop(index, axis = 0, inplace = True)
  # Déjà fait
    
    categorical_columns = []
    for index,col in enumerate(clean_data.columns): 
        if is_bool(clean_data[col]):
            pass
        elif is_numerical(clean_data[col]):
            clean_data[col].fillna(clean_data[col][0:1000].median(), inplace = True) 
            pass
          # Remplace les NaN par la médiane des 1000 premiers éléments
            clean_data[col] = (clean_data[col] -clean_data[col][0:1000].mean())/ clean_data[col][0:1000].std()
        # Normalise
        else:
            categorical_columns.append(str(datas.columns[index]))
            clean_data[col].apply(change_cell)

    clean_data = clean_data.append(datas.iloc[0], ignore_index=True)
    clean_data = pd.get_dummies(clean_data, columns=categorical_columns, drop_first=True)
  # Convert categorical variable into dummy/indicator variables. Generates binary vectors for each categorical col
  # in df. Removes redundance ie: each out col will correspond to unique category. Numerical and boolean unchanged.
  # 1. clean_data: Data of which to get dummy indicators. 
  # 2. columns=columns_categorical: Column names in the DataFrame to be encoded. 
  # in k diff col -> out k lines (0...k-1)
  # 3. drop_first=True: Get k-1 dummies out of k categorical levels by removing the first level.
  # in k diff col -> k lines (0,...k-2), 
    datas["energy_consumption_per_annum"] = labels["energy_consumption_per_annum"] # Ajoute energy à la fin de datas
    return clean_data


# In[205]:


x_train = cleandata(0, 10)
x_test  = cleandata(10, 20)
a = np.zeros(len(x_train))
for col in x_train.columns :
    try :
        x_test.col 
    except Exception as e : 
        x_test[col] = a
for col in x_test.columns :
    try :
        x_train.col 
    except Exception as e : 
        x_train[col] = a
        
        

print(x_train.shape, x_test.shape)


# In[206]:


x_train.head()


# In[207]:


x_test.head()


# In[208]:


import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
clean.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in clean.columns.values]
clean.columns.values # Converts to Object array and removes [ ] when necessary


# In[ ]:


clean.head()


# In[21]:


y_train = clean["energy_consumption_per_annum"]
X_train = clean.drop(["energy_consumption_per_annum"],axis=1)
# axis = 0 or index, 1 or columns -> Whether to drop labels from index or columns.


# In[35]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'subsample': [0.45, 0.5, 0.55], 'n_estimators': [1200, 1400], 
     'max_depth': [5], 'learning_rate': [0.02],
     'colsample_bytree': [0.4], 'colsample_bylevel': [0.5],
     'reg_alpha':[1], 'reg_lambda': [1], 'min_child_weight':[2]}
]

# n estimators : plus il est élevé plus il y a d'overfit. 100 pas bcp, 200 ou 300 bien, 1200 -> 1
# 

xgb_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)


# In[ ]:


xgb_model.fit(X_train, y_train)


# In[ ]:


score = xgb_model.score(X_train, y_train)
print(score)


# In[ ]:


grid_search = GridSearchCV(
    xgb_model,
    param_grid=param_grid,
    scoring = 'neg_root_mean_squared_error',
    n_jobs = -1,
    cv = 5,
    verbose=True
)


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:


print(grid_search.best_score_)


# In[ ]:


grid_search.best_estimator_.score(X_train,y_train)


# # code ancien

# In[ ]:


# Remove row 10 building period redondance avec building year
# Remove row 12 building type code (res/com) car redondance (row 12 après delete)
# Cast row 13 to int (1973.5)

datas.replace('', str(np.nan), inplace=True)
datas.replace([], str(np.nan), inplace=True)
datas.replace('[]', str(np.nan), inplace=True)
datas.replace('[NULL]', str(np.nan), inplace=True)
datas.replace(np.nan, str(np.nan), inplace=True)
datas.drop(["building_period", "building_use_type_code", "is_crossing_building"], axis=1, inplace=True)

datas.head(50)


# In[ ]:


"""
def is_nan(val_array):
    c = val_array.eq("nan").sum()
    return c

def is_date(date_str):
    format = "%Y-%m-%d"
    return bool(datetime.datetime.strptime(date_str, format))

def is_negative(col_nums, row):
    c=0
    for num in col_nums:
        if row[num] != 'nan' and row[num] <= 0:
            c+=1
    return c

def is_snegative(col_nums, row):
    c=0
    for num in col_nums:
        if row[num] != 'nan' and row[num] < 0:
            c+=1
    return c

def is_not_bool(col_nums, row):
    c=0
    for num in col_nums:
        if (row[num] != 'nan') and (not isinstance(row[num], bool)): 
            c += 1
    return c

def not_in_lmhvh(col_nums, row):
    c=0
    for num in col_nums:
        if row[num] != 'nan' and (row[num] not in ["low", "medium", "high", "very high"]):
            c += 1
    return c

def enough_data(row):
    counter = 0

    counter += is_nan(row)

    if (row["area_code"] != 'nan' and (row["area_code"] < 0 or row["area_code"] > 999)): 
        counter += 1
    
    if row["bearing_wall_material"] != 'nan' and (row["bearing_wall_material"] == "INDETERMINED"): 
        counter += 1
    
    
    counter += is_negative(["altitude", "building_height_ft", "building_total_area_sqft", "living_area_sqft", "lowe_floor_thermal_conductivity", "window_heat_retention_factor", "window_thermal_conductivity"], row)
    
    counter += is_snegative(["nb_commercial_units", "nb_dwellings", "nb_gas_meters_commercial", "nb_gas_meters_housing", "nb_gas_meters_total", "nb_housing_units", "nb_meters", "nb_parking_spaces", "nb_power_meters_commercial", "nb_power_meters_housing", "nb_power_meters_total", "nb_units_total", "outer_wall_thermal_conductivity", "upper_floor_thermal_conductivity", "percentage_glazed_surfaced"], row) 

    if row["building_year"] != 'nan' and (row["building_year"] <= 1300 or row["building_year"] >= 2023):
        counter += 1
    
    counter += not_in_lmhvh(["clay_risk_level", "radon_risk_level", "thermal_inertia"], row)
    
    if row["consumption_measurement_date"] != 'nan' and not is_date(row["consumption_measurement_date"]): 
        counter += 1
    
    counter += is_not_bool(["has_air_conditioning", "has_balcony", "solar_heating", "solar_water_heating"], row) 
    
    if row["heating_type"] != 'nan' and row["heating_type"] not in ['individual' 'collective']: 
        counter += 1
    
    if row["lower_floor_adjacency_type"] != 'nan' and row["lower_floor_adjacency_type"] not in ['FULL_EARTH', 'LNC', 'SANITARY_VACTILE', 'OUTSIDE', 'ADJACENT BUILDING', 'BURIED_WALL']:
        counter += 1
    
    if row["upper_floor_adjacency_type"] != 'nan' and row["upper_floor_adjacency_type"] not in ['EXTERNAL', 'LNC', 'ADJACENT_BUILDING', 'INNER_WALL']: 
        counter += 1
    
    if row["water_heating_type"] != 'nan' and row["water_heating_type"] not in ['individual', 'collective']: 
        counter += 1
    if counter >=10 : 
        datas.drop([i], inplace = True)
"""


# In[ ]:


counter=0
from tqdm import tqdm
tqdm(datas.apply(enough_data, axis = 0))


# In[ ]:


#Nous remplaçons toutes les valeures nulles par la même chaîne de caractères afin de pouvoir les manipuler 
datas.replace('', 'nan', inplace = True)
datas.replace([], 'nan', inplace = True)
datas.replace('[]', 'nan', inplace = True)
datas.replace('NULL', 'nan', inplace = True)

for col in datas:
    print('###############################################################')
    print(col)
    print(datas[col].unique())
# In[ ]:


### Définissons une fonction qui permet de traiter les valeurs en str en comptant les occurences de chaque terme

L = []
def format_col(col, Values):
    L = []
    for c in datas[col]:
        a = np.zeros(len(Values))
        for i in range(len(Values)) : 
            print(c)
            print(c.count(Values[i]))
            b = c.count(Values[i])
            a[i] = b
        L.append(a)
    datas[col] = L


# In[ ]:


To_formate_manually = ['building_category', 'building_class', 'main_heating_type', 'main_water_heating_type' ]
for column in datas.columns[1:] : 
    if (datas['{}'.format(column)].dtype != str):
        datas['{}'.format(column)] = datas['{}'.format(column)].astype(str)
        enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        X = datas['{}'.format(column)] 
        X = np.array(X).reshape(-1,1)
        enc.fit(X)
        datas['{}'.format(column)] = list(enc.transform(X))
format_col('building_category', ['condo', 'individual house'])
format_col('building_class', ['2 to 11', 'individual', '12+'])
format_col('main_heating_type', ['standard gas boiler', 'low temperature gas boiler', 'gas condensing boiler', 'electric boiler', 'Joule effect generators', 'solar heating'])
format_col('main_water_heating_type', ['low temperature oil boiler','lpg/butane/propane condensing boiler','joule-effect electric water heater','solar hot water','low temperature gas boiler','gas condensing boiler'])

###datas.head()


# In[ ]:


datas.to_csv('out.csv', index=False)


# In[ ]:


32.
### Here are some tests 

def isnan(array_value): 
    c=0
    for val in array_value :
        if (val.isnan()) :
            C+=1
    return c
def is_date(date_str):
    format = "%d-%m-%Y"
    return bool(datetime.strptime(date_str, format))

def enough_data(row):
    counter=0
    counter+=is_nan(row)
 
    if row[3] != nan && row[3] <= 0:  # altitude, pbp
        counter += 1
    if row[4] != nan && (row[4] < 0 || row[4]) > 999: # area code, pbp
        counter += 1
    if row[6] != nan && (row[6] == "INDETERMINED" || row[6]) == "OTHER": # bearing_wall_material
        counter += 1
    if row[9] != nan && row[9] <= 0: # building height, pbp
        counter += 1
    if row[10] != nan && row[10] <= 0: # building total area sqft, pbp
        counter += 1
    if row[13] != nan && (row[13] <= 1300 || row[13] >= 2023): # building year
        counter += 1
    if row[14] != nan && (row[14] not in ["low",medium",high"]):
        counter += 1
    if row[15] != nan && not(is_date(row[15])): # consumption_measurement_date
        counter += 1
    if row[16] != nan && not isinstance(row[16], bool): # has_air_conditioning
        counter += 1
    if row[17] != nan && not isinstance(row[16], bool): # has_balcony
        counter += 1
    if row[20] != nan && row[20] not in ['individual' 'collective']: # heating type
        counter += 1
    
    
    


# In[ ]:


### Standardising datas

for column in datas.columns : 
    if (column != 'level_0'):
        datas['{}'.format(column)] -= datas['{}'.format(column)].mean()
        datas['{}'.format(column)] /= datas['{}'.format(column)].std()
labels['energy_consumption_per_annum'] -= labels['energy_consumption_per_annum'].mean()
labels['energy_consumption_per_annum'] /= labels['energy_consumption_per_annum'].std()


# In[ ]:


### Merging datas to shuffle
datas['energy_consumption_per_annum'] = labels['energy_consumption_per_annum']


# In[ ]:


### On drop les ID qui n'apportent pas d'information sur la consommation

datas = datas.drop('level_0', axis = 1)

### On shuffle pour bien mélanger les données

shuffle(datas)

### On récupère les labels dans le nouvel ordre 

labels['energy_consumption_per_annum'] = datas['energy_consumption_per_annum']

### On enlève les labels des données d'entraînement et on récupère X et y dans des vecteurs 

datas = datas.drop('energy_consumption_per_annum', axis = 1)
y_train = labels['energy_consumption_per_annum']
X_train = np.array(datas.transpose()).reshape(1010684,70,1)


# In[ ]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint 
import sklearn
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


# In[ ]:


model = Sequential()

model.add(LSTM(128,input_shape=(X_train.shape[1:]), return_sequences = True, activation = 'tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences = True, activation = 'tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation = 'tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation = 'softmax'))


# In[ ]:


opt = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 1e-6)
model.compile(loss = 'mse', optimizer = opt, metrics = ['accuracy'])


# In[ ]:


call_list = [tf.keras.callbacks.EarlyStopping(patience = 100)]

History = model.fit(X_train, y_train, 
                    validation_data = (X_test, y_test)
                   batch_size = 32,
                   epochs = 10,
                   callbacks = call_list )


# In[ ]:


y_pred = model.predict(X_test)
Explained_variance = sklearn.metrics.explained_variance_score
print(Explained_variance)


# In[ ]:


### Ici on peut tenter une PCA
from sklearn.decomposition import PCA 
pca = PCA(n_components = 10)
pca.fit(datas)
x_pca = pca.transform(datas)


# In[ ]:


y_train = labels['energy_consumption_per_annum']
X_train = np.array(datas.transpose()).reshape(1010684,70)


# In[ ]:




# First, split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Next, create an instance of the RandomForestRegressor class
# you can specify the number of trees in the forest with the n_estimators parameter
# you can specify the number of features to consider at each split with max_features parameter
# you can set the random_state to get reproducible results
rf = RandomForestRegressor(n_estimators=100, max_features='sqrt',random_state=0)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Use the model to predict on the testing data
y_pred = rf.predict(X_test)

# Evaluate the model's performance using metrics such as mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# First, split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Next, create an instance of the RandomForestRegressor class
rf = RandomForestRegressor(n_estimators=100, max_features='sqrt',random_state=0)

rf.fit(X_train, y_train)
 
# Use the model to predict on the testing data
y_pred = rf.predict(X_test)

# Evaluate the model's performance using metrics such as mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)


# In[ ]:


'[condo,condo,condo,condo,condo,condo,condo,condo,condo,condo]'.count('condo')


# In[ ]:



