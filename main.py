# Librerías
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('bmh')
sns.set_style("whitegrid")

import os 
os.environ['TCL_LIBRARY'] = 'C:/Users/Alfres/AppData/Local/Programs/Python/Python313/tcl/tcl8.6'
os.environ['TK_LIBRARY'] = 'C:/Users/Alfres/AppData/Local/Programs/Python/Python313/tcl/tk8.6'

# Set de opciones de pandas
pd.set_option("display.max_rows", None)

# Inicializamos el escalador
sc = StandardScaler()

# Se define el nombre de las columnas
index_names = ['motor', 'ciclos_operacion']
setting_names = ['configuracion_1', 'configuracion_2', 'configuracion_3']
sensor_names = ['sensor_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names

# Se cargan los datos
train = pd.read_csv('CMAPSSData\\train_FD001.txt', sep='\\s+', header=None, names=col_names)
test = pd.read_csv('CMAPSSData\\train_FD001.txt', sep='\\s+', header=None, names=col_names)
y_test = pd.read_csv('CMAPSSData\\RUL_FD001.txt', sep='\\s+', header=None, names=['RUL'])

# CORRECCIONES: Filtrar test para que coincida con los motores en y_test
# Agrega un índice de motor al DataFrame y_test para alinear los datos
y_test['motor'] = range(1, 101)  # Motores del 1 al 100
test = test[test['motor'].isin(y_test['motor'])]

# Asegúrate de que test esté ordenado igual que y_test
test = test.sort_values(by='motor')
y_test = y_test.sort_values(by='motor')

# Función para agregar RUL al conjunto de datos
def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="motor")
    max_cycle = grouped_by_unit["ciclos_operacion"].max()
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='motor', right_index=True)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["ciclos_operacion"]
    result_frame["RUL"] = remaining_useful_life
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

# Calculamos el RUL para el conjunto de entrenamiento
train = add_remaining_useful_life(train)
train[sensor_names + ['RUL']].head()

# Histograma de RUL para los motores
df_max_rul = train[['motor', 'RUL']].groupby('motor').max().reset_index()
df_max_rul['RUL'].hist(bins=15, figsize=(15,7))
plt.xlabel('RUL')
plt.ylabel('Motores')
plt.show()

# Función para graficar sensores
def plot_sensor(sensor_name):
    plt.figure(figsize=(13, 5))
    for i in train['motor'].unique():
        if (i % 10 == 0):  # Selecciona motores divisibles por 10
            plt.plot('RUL', sensor_name, data=train[train['motor'] == i], label=f'Motor {i}')
    plt.xlim(250, 0)  
    plt.xticks(np.arange(0, 275, 25))
    plt.ylabel(sensor_name)
    plt.xlabel('Remaining Useful Life')
    plt.legend(loc='best')
    plt.title(f'Sensor: {sensor_name}')
    plt.show()

# Generar gráficos para cada sensor
for sensor_name in sensor_names:
    plot_sensor(sensor_name)

# Correlación entre variables
cor = train.corr()
train_relevant_features = cor[abs(cor['RUL']) >= 0.5]
train_relevant_features['RUL']

list_relevant_features = train_relevant_features.index
list_relevant_features = list_relevant_features[1:]  # Eliminamos la columna 'RUL' de la lista de características
train = train[list_relevant_features]

# División en X_train y y_train
y_train = train['RUL']
X_train = train.drop(['RUL'], axis=1)

# Filtrar y preparar datos de test
X_test = test[X_train.columns]  # Usar las mismas características seleccionadas en X_train

# Calcular el RUL para el conjunto de test
test = add_remaining_useful_life(test)
y_test = test[['motor', 'RUL']]  # Tomamos la columna de RUL calculada en test

# Asegúrate de que test y y_test estén alineados
y_test = y_test.sort_values(by='motor')
test = test.sort_values(by='motor')

# Escalar los datos
X_train1 = sc.fit_transform(X_train)
X_test1 = sc.transform(X_test)

# Función para evaluar el modelo
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE: {}, R2: {}'.format(label, rmse, variance))
    return rmse, variance

# Modelo XGBoost
xgb_r = xg.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
xgb_r.fit(X_train1, y_train)

# Evaluar en el conjunto de entrenamiento
y_hat_train1 = xgb_r.predict(X_train1)
RMSE_Train, R2_Train = evaluate(y_train, y_hat_train1, 'train')

# Evaluar en el conjunto de prueba
y_hat_test1 = xgb_r.predict(X_test1)
RMSE_Test, R2_Test = evaluate(y_test['RUL'], y_hat_test1)

# Resultados finales
Results = pd.DataFrame({
    'Model': ['XGBoost'],
    'RMSE-Train': [RMSE_Train],
    'R2-Train': [R2_Train],
    'RMSE-Test': [RMSE_Test],
    'R2-Test': [R2_Test]
})

print(Results)
Results
