from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor, Lasso, Ridge, RidgeClassifier
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import pandas as pd
from statistics import mean

df = pd.read_csv('sunflare_dataset.txt')
df.loc[:, ['activity', 'evolution', 'previous', 'history', 'complex', 'area', 'area_largest']] -= 1

df['activity'] = df['activity'].apply(str)
df['evolution'] = df['evolution'].apply(str)
df['previous'] = df['previous'].apply(str)
df['history'] = df['history'].apply(str)
df['complex'] = df['complex'].apply(str)
df['area'] = df['area'].apply(str)
df['area_largest'] = df['area_largest'].apply(str)
# df['C_flares'] = df['C_flares'].apply(str)
# df['M_flares'] = df['M_flares'].apply(str)
# df['X_flares'] = df['X_flares'].apply(str)

cat = df.drop(['C_flares', 'M_flares', 'X_flares'], 1)

x_cat = cat.to_dict(orient='records')

vec = DictVectorizer(sparse=False)
vec_x_cat = vec.fit_transform(x_cat)
# print(vec.feature_names_)

X = np.array(vec_x_cat)
# y = np.array(df[['C_flares', 'M_flares', 'X_flares']])
y = np.array(df[['C_flares']]).ravel()

accuracies = []


def run():

    # poly = preprocessing.PolynomialFeatures(degree=2)
    # x_quad = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    nn = MLPRegressor(
        hidden_layer_sizes=(50, 50),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=False,
        random_state=80, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # nn = LogisticRegression(n_jobs=-1, solver='liblinear')
    # nn = LinearRegression(n_jobs=-1)

    # nn = MLPClassifier(hidden_layer_sizes=(400, 400, 400, 400, 100), tol=0.0001, solver='adam', activation='relu', random_state=54)

    # nn = RandomForestClassifier()
    # nn = KNeighborsClassifier()
    # nn = SGDClassifier()
    # nn = LinearSVC()
    # nn = SVC(kernel='rbf')
    # nn = RidgeClassifier()
    # nn = GaussianNB()

    nn.fit(X_train, y_train)
    accuracy = nn.score(X_test, y_test)
    print(accuracy)
    accuracies.append(accuracy)

for _ in range(1):
    run()

print()
print(mean(accuracies))
