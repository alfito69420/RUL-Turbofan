
#   Librerías 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
plt.style.use('bmh')
sns.set_style("whitegrid")

import os 
os.environ['TCL_LIBRARY'] = 'C:/Users/Alfres/AppData/Local/Programs/Python/Python313/tcl/tcl8.6'
os.environ['TK_LIBRARY'] = 'C:/Users/Alfres/AppData/Local/Programs/Python/Python313/tcl/tk8.6'

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option("display.max_rows", None)

# se define el nombre de las columnas
index_names = ['motor', 'ciclos_operacion']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names

#se cargan los datos
train = pd.read_csv('CMAPSSData\\train_FD001.txt',sep='\\s+', header=None, names=col_names)
test = pd.read_csv('CMAPSSData\\train_FD001.txt',sep='\\s+', header=None, names=col_names)
y_test = pd.read_csv('CMAPSSData\\RUL_FD001.txt', sep='\\s+', header=None, names=['RUL'])

# El archivo train contiene todas las caracteristicas como numero de unidad, parametros de configuracion y sensores
# El archivo test contiene todas las caracteristicas como numero de unidad, parametros de configuracion y sensores
# El archivo Y_test contiene el RUL para los datos de test
train.head()
print(train.head())


train['motor'].unique() # Hay 100 motores no unicos
y_test.shape    # RUL value for 100 no of engines 
(100,1)
print(train.describe())

train=train.drop('setting_3',axis=1)

def add_remaining_useful_life(df):
    #   Obten el numero total de ciclos por cada unidad
    grouped_by_unit = df.groupby(by="motor")
    max_cycle = grouped_by_unit["ciclos_operacion"].max()
    
    #   Fusionar el ciclo máximo nuevamente en el marco original
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='motor', right_index=True)
    
    #   Calcula el RULL de cada fila
    remaining_useful_life = result_frame["max_cycle"] - result_frame["ciclos_operacion"]
    result_frame["RUL"] = remaining_useful_life
    
    #Se elimina max_cycle, ya no se ocupa
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

train = add_remaining_useful_life(train)
train[sensor_names+['RUL']].head()

df_max_rul = train[['motor', 'RUL']].groupby('motor').max().reset_index()
df_max_rul['RUL'].hist(bins=15, figsize=(15,7))
plt.xlabel('RUL')
plt.ylabel('frequency')
plt.show()