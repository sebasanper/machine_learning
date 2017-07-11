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
import pickle

# df = pd.read_csv('coords3x3_full_copy.dat')
#
# with open("pickle_full_data.pickle", 'wb') as out:
#     pickle.dump(df, out)

df = pickle.load(open('pickle_full_data.pickle', 'rb'))
training_set = df.sample(frac=0.9, replace=True)
df = df.sample(frac=0.005, replace=True)
"""frac = 0.004 for estimating 'aep'
   frac = 0.001 for estimating 'lcoe'"""
# print(df.head())

# df['activity'] = df['activity'].apply(str)

numeric_cols = ['nbins', 'dir_real', 'dir_artif']

x_num = df[numeric_cols].as_matrix()
x_num_training = training_set[numeric_cols].as_matrix()

# max_x = np.amax(x_num, 0)
#
# x_num = x_num / max_x

cat = df.drop(numeric_cols + ['aep', 'lcoe', 'power_calls', 'ct_calls'], 1)
cat_training = training_set.drop(numeric_cols + ['aep', 'lcoe', 'power_calls', 'ct_calls'], 1)

x_cat = cat.to_dict(orient='records')
x_cat_training = cat_training.to_dict(orient='records')

vec = DictVectorizer(sparse=False)
vec_x_cat = vec.fit_transform(x_cat)
vec_x_cat_training = vec.fit_transform(x_cat_training)
X = np.hstack((x_num, vec_x_cat))
X_training = np.hstack((x_num_training, vec_x_cat_training))
# print(vec.feature_names_)

y = np.array(df[['time']]).ravel()
y_training = np.array(training_set[['time']]).ravel()

accuracies = []


def run():

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    nn = MLPRegressor(
        hidden_layer_sizes=(100, 100),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=False,
        random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
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
    print(len(X_train), len(y_train))
    nn.fit(X_train, y_train)
    print(len(X_test), len(y_test))
    accuracy = nn.score(X_test, y_test)
    print('small accuracy:', accuracy)
    print(len(X_training), len(y_training))
    print('big accuracy:', nn.score(X_training, y_training))
    accuracies.append(accuracy)

for _ in range(1):
    run()

print()
print('mean accuracy is:', mean(accuracies))
