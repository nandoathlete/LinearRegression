import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.construct import random
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

## REGRESION LINEAL SIMPLE ##
# Y = mX + b + e


# Cargando conjunto de datos
plt.style.use('ggplot')
Datos = pd.read_csv('Salary.csv')
print("DATOS:\n ", Datos)
print("Tamaño del Datagrama")
print(f"Fila: {Datos.shape[0]}")
print(f"Columna: {Datos.shape[1]}")
X = Datos['YearsExperience']
Y = Datos ['Salary']
# Visualizando conjunto de Datos
plt.figure(1)
plt.scatter(X, Y, color = 'red', marker = '.')
plt.title("Years Experience vs Salary")
plt.xlabel('Year Experience')
plt.ylabel('Salary')

# Analizando los datos
if Y.isnull().sum():
    print("Los Datos del Salario Contiene Datos Nulos")
elif X.isnull().sum():
    print("Los Datos Años de Experiencia Contiene Datos Nulos")
else:
    print("Sin datos nulos")

# Que tan correlacionadas están las variables
Correlacion = np.corrcoef(X,Y)
plt.figure(2)
ax = plt.axes()
sns.heatmap(data = Correlacion, annot = True)
ax.set_title('Correlación de Pearson')

# Dividir Datos de Entrenamiento y Prueba
# * Ley de Paretto 80/20
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, shuffle = True, random_state = 42)

# Creando modelo de Regresión Lineal
# * Pasando Arrays de 1D a 2D
X_train = X_train.values.reshape(-1,1) # --> Para el modelo
X_test = X_test.values.reshape(-1,1) # --> Para Prediccion
Modelo_RL = LinearRegression().fit(X_train, y_train)

# Realizando predicción
y_test_predict = Modelo_RL.predict(X_test)

# Evaluando el modelo con r2_score: (Datos Reales de Prueba, Datos Reales de la Predicción)
R2 = r2_score(y_test, y_test_predict)
print(f"R2: {R2}")

# Calculando pendiente y punto de corte: y = mx + b
print(f"Coeficiente: {Modelo_RL.coef_}") # -> m (Pendiente)
print(f"Punto de corte: {Modelo_RL.intercept_}")

# Recta de Regresion Lineal (Visualizando)
X_YearsExperience = X.values.reshape(-1,1)
Y_YearsExperiencePrediction = Modelo_RL.predict(X_YearsExperience)
plt.figure(3)
plt.scatter(X, Y, color = 'blue', marker = '.', label = 'Datos Reales')
plt.plot(X, Y_YearsExperiencePrediction, color = 'black', linestyle = '--', label = 'Prediction')
plt.title('YearsExperience vs Salary')
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.legend()
#plt.show()

# ---------------------------------------------------------------
# Insetando año de experiencia personal:
print("\n\n\n")
Experiencia = int(input("Años de experiencia personal: "))
Prediccion = Modelo_RL.predict(np.array([[Experiencia]]))
print("Salario estimado: ", Prediccion)