from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
data_total = pd.read_csv(url)
data_total.head()

data_total.to_csv('/workspaces/Gonz3092-EDA-proyecto-4geeks/data/raw/data_total.csv')

data_total.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)

data_total["room_type_n"] = pd.factorize(data_total["room_type"])[0]
data_total["neighbourhood_group_n"] = pd.factorize(data_total["neighbourhood_group"])[0]
data_total["neighbourhood_n"] = pd.factorize(data_total["neighbourhood"])[0]

data_total = data_total[data_total["price"] > 0]

data_total = data_total[data_total["minimum_nights"] <= 15]

data_total = data_total[data_total ["calculated_host_listings_count"] > 4]

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", "availability_365", "neighbourhood_group_n", "room_type_n"]

x = data_total.drop("price", axis=1)[num_variables]
y = data_total['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()

x_train_norm = pd.DataFrame(scaler.fit_transform(x_train), columns=num_variables)
x_test_norm = pd.DataFrame(scaler.transform(x_test), columns=num_variables)

selection_model = SelectKBest(f_classif, k = 5)
selection_model.fit(x_train, y_train)
ix = selection_model.get_support()
x_train_sel = pd.DataFrame(selection_model.transform(x_train), columns = x_train.columns.values[ix])
x_test_sel = pd.DataFrame(selection_model.transform(x_test), columns = x_test.columns.values[ix])

x_train_sel["price"] = list(y_train)
x_test_sel["price"] = list(y_test)
x_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
x_test_sel.to_csv("../data/processed/clean_test.csv", index = False)