# %% [markdown]
# # New York Airbnb EDA

# %% [markdown]
# ## Paso 1: Recopilación de datos

# %%
import pandas as pd

total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")

total_data.head()

# %%
# Definir la ruta donde se guardará el DataFrame en bruto

ruta_data_frame_bruto = r"C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-machine-learning-python-template/data/raw/total_data.csv"

# Guardar el DataFrame en formato CSV

total_data.to_csv(ruta_data_frame_bruto, index=False, encoding='utf-8')

# %%
# Crear una copia del DataFrame original

interim_data = total_data.copy()

# Verificar que la copia se ha realizado correctamente

interim_data.head()

# %%
# Definir la ruta donde se guardará el DataFrame de datos intermedios

ruta_data_frame_intermedio = r"C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-machine-learning-python-template/data/interim/interim_data.csv"

# Guardar el DataFrame en formato CSV

total_data.to_csv(ruta_data_frame_intermedio, index=False, encoding='utf-8')

# %% [markdown]
# ## Paso 2: Exploración y limpieza de datos

# %%
# Obtener las dimensiones

interim_data.shape

# %% [markdown]
# El DataFrame contiene 48.895 registros (filas) y 16 variables (columnas)

# %%
# Obtener información sobre tipos de datos y valores no nulos

interim_data.info()

# %% [markdown]
# 🏗 **Consideraciones para Preprocesamiento**
# 
# - **Variables categóricas (`object`)**: `name`, `host_name`, `neighbourhood_group`, `neighbourhood`, `room_type`, `last_review`.
# 
# - **Variables numéricas (`int64`, `float64`)**: `price`, `minimum_nights`, `reviews_per_month`, `latitude`, `longitude`, etc.
# 
# - **Datos nulos importantes**:
#   
#   - `last_review` y `reviews_per_month` tienen **más de 10,000 valores faltantes**, lo que representa aproximadamente el **20% de los datos**.
#   
#   - `name` y `host_name` tienen pocos valores nulos y podrían ser completados o descartados según el análisis.

# %% [markdown]
# ### Eliminar duplicados

# %%
print(f"El número de duplicados de registros ID es: {interim_data['id'].duplicated().sum()}")
print(f"El número de duplicados de registros Name es: {interim_data['name'].duplicated().sum()}")
print(f"El número de duplicados de registros Host ID es: {interim_data['host_id'].duplicated().sum()}")

# %% [markdown]
# No se eliminan duplicados porque cada registro corresponde a un ID único.

# %% [markdown]
# ### Eliminar información irrelevante

# %%
interim_data.drop(["id", "name", "host_id", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)

interim_data.head()

# %% [markdown]
# ## Paso 3: Análisis de variables univariante

# %% [markdown]
# ### Análisis sobre variables categóricas

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axis = plt.subplots(2, 2, figsize=(14, 11))  # Aumentar tamaño del gráfico

# Crear histogramas
sns.histplot(ax=axis[0,0], data = interim_data, x = "neighbourhood_group").set(ylabel=None)
sns.histplot(ax=axis[0,1], data = interim_data, x = "neighbourhood").set_xticks([])
sns.histplot(ax=axis[1,0], data = interim_data, x = "room_type").set(ylabel=None)
sns.histplot(ax=axis[1,1], data = interim_data, x = "availability_365").set(ylabel=None)

# Rotar etiquetas en cada gráfico
for ax in axis.flatten():  
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")  # Rotar etiquetas del eje x

plt.tight_layout()  # Ajustar layout para evitar superposición
plt.show()

# %% [markdown]
# - Tenemos 5 aréas o distritos: Brooklyn, Manhattan, Queens, Staten Island y Bronx. La mayoría de las ofertas se concentran en Manhattan y en Brooklyn.
# - Hay más casas/apartamentos disponibles, que habitaciones privadas. Las habitaciones compartidas es la categorìa con menos frecuencia.
# - La mayorìa de las habitaciones están disponibles los 365 días del año; sin embargo, hay muchos 0/Nan lo que puede sugerir un error o falta de información.

# %% [markdown]
# 🏗 **Consideraciones para Preprocesamiento**
# 
# - **Variables categóricas (`object`)**: `neighbourhood_group`, `neighbourhood`, `room_type`, `availability_365`.
# 
# - **Variables numéricas (`int64`, `float64`)**: `price`, `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count`.

# %% [markdown]
# ### Análisis de variables númericas

# %%
fig, axis = plt.subplots(4, 2, figsize=(10, 14), gridspec_kw = {"height_ratios": (6, 1, 6, 1)})

# Crear histogramas y boxplots

sns.histplot(ax=axis[0,0], data = interim_data, x = "price")
sns.boxplot(ax=axis[1,0], data = interim_data, x = "price")

sns.histplot(ax=axis[0,1], data = interim_data, x = "minimum_nights")
sns.boxplot(ax=axis[1,1], data = interim_data, x = "minimum_nights")

sns.histplot(ax=axis[2,0], data = interim_data, x = "number_of_reviews")
sns.boxplot(ax=axis[3,0], data = interim_data, x = "number_of_reviews")

sns.histplot(ax=axis[2,1], data = interim_data, x = "calculated_host_listings_count")
sns.boxplot(ax=axis[3,1], data = interim_data, x = "calculated_host_listings_count")

plt.tight_layout()  # Ajustar layout para evitar superposición
plt.show()

# %% [markdown]
# ## Paso 4: Análisis de variables multivariantes

# %% [markdown]
# ### Análisis numérico-numérico

# %% [markdown]
# *Price - (Minimum_nights, Number_of_reviews, Calculated_host_listings_count)*

# %%
import seaborn as sns

# Configuración de estilo
sns.set(style="whitegrid")

# Crear figura y ejes
fig, axis = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

# ---------------------- SCATTERPLOTS ----------------------
sns.regplot(ax=axis[0, 0], data=interim_data, x="minimum_nights", y="price")
axis[0, 0].set_title("Minimum Nights vs Price")

sns.regplot(ax=axis[0, 1], data=interim_data, x="number_of_reviews", y="price")
axis[0, 1].set_title("Number of Reviews vs Price")

sns.regplot(ax=axis[0, 2], data=interim_data, x="calculated_host_listings_count", y="price")
axis[0, 2].set_title("Listings per Host vs Price")

# ---------------------- HEATMAPS ----------------------
sns.heatmap(interim_data[["price", "minimum_nights"]].corr(), 
            annot=True, fmt=".2f", ax=axis[1, 0], cbar=False, cmap="coolwarm", square=True)
axis[1, 0].set_title("Correlation")

sns.heatmap(interim_data[["price", "number_of_reviews"]].corr(), 
            annot=True, fmt=".2f", ax=axis[1, 1], cbar=False, cmap="coolwarm", square=True)
axis[1, 1].set_title("Correlation")

sns.heatmap(interim_data[["price", "calculated_host_listings_count"]].corr(), 
            annot=True, fmt=".2f", ax=axis[1, 2], cbar=False, cmap="coolwarm", square=True)
axis[1, 2].set_title("Correlation")

# Rotar etiquetas de los heatmaps solo si es necesario
for ax in axis[1]:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

# Ajustar diseño
plt.tight_layout(h_pad=3)
plt.show()

# %% [markdown]
# - No hay relación entre el precio y el número mínimo de noches.
# - No hay relación entre el precio y el número de reviews.
# - No hay relación entre el precio y el recuento de listados de host calculados.

# %% [markdown]
# ### Análisis categórico-categórico

# %% [markdown]
# *Room_type - Neighbourhood group*

# %%
fig, axis = plt.subplots(figsize=(5, 4))

sns.countplot(data=interim_data, x="room_type", hue="neighbourhood_group")

plt.title("Room Type by Neighbourhood Group")

plt.show()

# %% [markdown]
# - Manhattan es el distrito que concentra el mayor número de lugares para alquilar, especificamente casas o apartamentos enteros.
# - Brooklyn es el segundo distrito con más lugares para alquilar, especialmente habitaciones privadas.
# - Queens y Bronx son el tercer y cuarto distrito, destacando en una mayor frecuencia en habitaciones privadas que en casas o apartamentos enteros.

# %% [markdown]
# ### Análisis numérico-categórico (completo)

# %%
# Factorizar variables categóricas
interim_data["neighbourhood_group"] = pd.factorize(interim_data["neighbourhood_group"])[0]
interim_data["neighbourhood"] = pd.factorize(interim_data["neighbourhood"])[0]
interim_data["room_type"] = pd.factorize(interim_data["room_type"])[0]

# Seleccionar las columnas a incluir en el análisis
columns_to_include = [
    "neighbourhood_group",
    "neighbourhood",
    "room_type",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "calculated_host_listings_count",
    "availability_365"
]

# Mapa de calor de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(interim_data[columns_to_include].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# %% [markdown]
# **room_type es la variable más correlacionada con price**
# 
# - Correlación de 0.21 → positiva, pero baja.
# - Implica que el tipo de habitación influye en el precio (como era de esperarse: habitaciones privadas vs. apartamentos enteros).
# - Aunque no es una relación fuerte, es la más significativa en este conjunto.

# %% [markdown]
# ### Analizar todos los datos simultáneamente

# %%
sns.pairplot(data = interim_data)

# %% [markdown]
# **Price vs otras variables:**
# - No se observa una relación lineal clara con ninguna variable.
# - Hay mucha dispersión y concentración en valores bajos, con algunos outliers muy por encima.
# - Esto refuerza lo visto en el heatmap: las correlaciones lineales con price son débiles.
# 
# **Multicolinealidad baja**
# - No se observan patrones diagonales o elípticos entre pares de variables (excepto lat-lon).
# - Esto sugiere que las variables no están muy correlacionadas entre sí → baja multicolinealidad.

# %% [markdown]
# ## Paso 5: Ingeniería de características

# %% [markdown]
# ### Análisis de outliers

# %%
interim_data.describe()

# %%
fig, axes = plt.subplots(2, 4, figsize=(12, 10))

sns.boxplot(ax=axes[0, 0], data=interim_data, y = "neighbourhood_group")
sns.boxplot(ax=axes[0, 1], data=interim_data, y = "neighbourhood")
sns.boxplot(ax=axes[0, 2], data=interim_data, y = "room_type")
sns.boxplot(ax=axes[0, 3], data=interim_data, y = "price")
sns.boxplot(ax=axes[1, 0], data=interim_data, y = "minimum_nights")
sns.boxplot(ax=axes[1, 1], data=interim_data, y = "number_of_reviews")
sns.boxplot(ax=axes[1, 2], data=interim_data, y = "calculated_host_listings_count")
sns.boxplot(ax=axes[1, 3], data=interim_data, y = "availability_365")

plt.tight_layout()
plt.show()

# %% [markdown]
# *Identificación de outliers para `price`*

# %%
# Resumen estadístico

price_stats = interim_data["price"].describe()
price_stats

# %%
# IQR para Price

IQR = price_stats["75%"] - price_stats["25%"]
Q3 = price_stats["75%"] + 1.5 * IQR
Q1 = price_stats["25%"] - 1.5 * IQR

print(f"The upper and lower limits are {round(Q3, 2)} and {round(Q1, 2)}, with an interquartile range of {round(IQR, 2)}")

# %%
# Contar cuántos valores de price son menores a 0

len(interim_data[interim_data["price"]<0])

# %%
# Contar cuántos valores de price son iguales a 0

len(interim_data[interim_data["price"]==0])

# %%
interim_data[interim_data["price"]==0]

# %%
# Contar cuántos valores de price son mayores a 334

len(interim_data[interim_data["price"]>334])

# %%
# Limpiar los valores atípicos

price_median = price_stats["50%"]

interim_data.loc[interim_data["price"] == 0, "price"] = price_median

# %%
count_0 = interim_data[interim_data["price"] == 0].shape[0]

print("Count of 0: ", count_0)

# %% [markdown]
# *Identificación de outliers para `minimum_nights`*

# %%
# Resumen estadístico

minimum_nights_stats = interim_data["minimum_nights"].describe()
minimum_nights_stats

# %%
# IQR para "minimum_nights"

IQR = minimum_nights_stats["75%"] - minimum_nights_stats["25%"]
Q3 = minimum_nights_stats["75%"] + 1.5 * IQR
Q1 = minimum_nights_stats["25%"] - 1.5 * IQR

print(f"The upper and lower limits are {round(Q3, 2)} and {round(Q1, 2)}, with an interquartile range of {round(IQR, 2)}")

# %%
# Contar cuántos valores de "minimum_nights" son mayores a 11

len(interim_data[interim_data["minimum_nights"]>11])

# %%
interim_data[interim_data["minimum_nights"]>11]

# %%
# Limpiar los valores atípicos

minimum_nights_median = minimum_nights_stats["50%"]

interim_data.loc[interim_data["minimum_nights"] > 15, "minimum_nights"] = minimum_nights_median

# %% [markdown]
# *Identificación de outliers para `number_of_reviews`*

# %%
# Resumen estadístico

number_of_reviews_stats = interim_data["number_of_reviews"].describe()
number_of_reviews_stats

# %%
# IQR para "number_of_reviews"

IQR = number_of_reviews_stats["75%"] - number_of_reviews_stats["25%"]
Q3 = number_of_reviews_stats["75%"] + 1.5 * IQR
Q1 = number_of_reviews_stats["25%"] - 1.5 * IQR

print(f"The upper and lower limits are {round(Q3, 2)} and {round(Q1, 2)}, with an interquartile range of {round(IQR, 2)}")

# %% [markdown]
# *Identificación de outliers para `calculated_host_listings_count`*

# %%
# Resumen estadístico

calculated_host_listings_count_stats = interim_data["calculated_host_listings_count"].describe()
calculated_host_listings_count_stats

# %%
# IQR para "calculated_host_listings_count"

IQR = calculated_host_listings_count_stats["75%"] - calculated_host_listings_count_stats["25%"]
Q3 = calculated_host_listings_count_stats["75%"] + 1.5 * IQR
Q1 = calculated_host_listings_count_stats["25%"] - 1.5 * IQR

print(f"The upper and lower limits are {round(Q3, 2)} and {round(Q1, 2)}, with an interquartile range of {round(IQR, 2)}")

# %%
# Contar cuántos valores de "calculated_host_listings_count" son mayores a 4

len(interim_data[interim_data["calculated_host_listings_count"]>4])

# %%
interim_data[interim_data["calculated_host_listings_count"]>4]

# %%
# Limpiar los valores atípicos

calculated_host_listings_count_median = calculated_host_listings_count_stats["50%"]

interim_data.loc[interim_data["calculated_host_listings_count"] > 4, "calculated_host_listings_count"] = calculated_host_listings_count_median

# %% [markdown]
# ### Inferencia de nuevas características

# %%
from geopy.distance import geodesic

# Coordenada central (Times Square, NYC)
ref_point = (40.7580, -73.9855)

interim_data["distance_to_center"] = interim_data.apply(
    lambda row: geodesic((row["latitude"], row["longitude"]), ref_point).km, axis=1
)

# %% [markdown]
# ### Análisis de valores faltantes

# %%
# Count NaN

interim_data.isnull().sum().sort_values(ascending = False) / len(interim_data)

# %% [markdown]
# ### Escalado de valores

# %%
from sklearn.model_selection import train_test_split

num_variables = ["neighbourhood_group", "neighbourhood", "room_type", "minimum_nights", "number_of_reviews", "calculated_host_listings_count", "availability_365", "distance_to_center"]

# Dividimos el conjunto de datos en muestras de train y test
X = interim_data.drop("price", axis = 1)[num_variables]
y = interim_data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()

# %% [markdown]
# *Normalización*

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_norm = scaler.transform(X_train)
X_train_norm = pd.DataFrame(X_train_norm, index = X_train.index, columns = num_variables)

X_test_norm = scaler.transform(X_test)
X_test_norm = pd.DataFrame(X_test_norm, index = X_test.index, columns = num_variables)

X_train_norm.head()

# %% [markdown]
# ## Paso 6: Selección de características

# %%
from sklearn.feature_selection import f_classif, SelectKBest

# Con un valor de k = 5 decimos implícitamente que queremos eliminar 3 características del conjunto de datos
selection_model = SelectKBest(f_classif, k = 5)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

# %%
X_train_sel.head()

# %%
X_test_sel.head()

# %% [markdown]
# ### Save the clean data

# %%
X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-machine-learning-python-template/data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-machine-learning-python-template/data/processed/clean_test.csv", index = False)


