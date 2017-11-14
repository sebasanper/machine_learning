from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor, Lasso, Ridge, RidgeClassifier
from sklearn import model_selection
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.feature_extraction import DictVectorizer
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
# import matplotlib.pyplot as plt
# from random import uniform
import pandas as pd
from statistics import mean, stdev
import pickle

df_original = pd.read_csv('coords3x3_full_copy.dat')
print(len(df_original))
# with open("pickle_full_data.pickle", 'wb') as out:
#     pickle.dump(df, out)

# df = pickle.load(open('pickle_full_data.pickle', 'rb'))

df_original['A'] = df_original['A'].apply(str)
df_original['B'] = df_original['B'].apply(str)
df_original['C'] = df_original['C'].apply(str)
df_original['D'] = df_original['D'].apply(str)
df_original['E'] = df_original['E'].apply(str)
df_original['F'] = df_original['F'].apply(str)
df_original['G'] = df_original['G'].apply(str)
df_original['H'] = df_original['H'].apply(str)
df_original['I'] = df_original['I'].apply(str)
df_original['J'] = df_original['J'].apply(str)

df = df_original.sample(frac=0.08, replace=False)
print(len(df))
bigtest_set = df_original.sample(frac=0.01, replace=False)
print(len(df))
"""frac = 0.004 for estimating 'aep'
   frac = 0.001 for estimating 'lcoe'"""


x_numeric_cols = ['nbins', 'dir_real', 'dir_artif']
y_numeric_cols = ['lcoe']

x_num = df[x_numeric_cols].as_matrix()
print(x_num[12])
y_num = df[y_numeric_cols].as_matrix()
# print(len(x_num))
x_num_bigtest = bigtest_set[x_numeric_cols].as_matrix()
y_num_bigtest = bigtest_set[y_numeric_cols].as_matrix()

# max_y = np.amax(df['lcoe'], 0)
max_y = 1.0
max_x = np.amax(df_original[x_numeric_cols].as_matrix(), 0)

# print(max_x)
x_num /= max_x
y_num /= max_y
x_num_bigtest /= max_x
y_num_bigtest /= max_y

cat = df.drop(x_numeric_cols + ['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1)

# print(len(cat))
cat_bigtest = bigtest_set.drop(x_numeric_cols + ['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1)

x_cat = cat.to_dict(orient='records')
x_cat_bigtest = cat_bigtest.to_dict(orient='records')

vec = DictVectorizer(sparse=False)
vec = vec.fit(x_cat_bigtest)
print(vec.feature_names_)

# with open("pickle_vectorizer.pickle", 'wb') as out:
#     pickle.dump(vec, out)

# vec = pickle.load(open('pickle_vectorizer.pickle', 'rb'))
print(x_cat_bigtest[0])
vec_x_cat_bigtest = vec.transform(x_cat_bigtest)
print(vec_x_cat_bigtest[0])
vec_x_cat = vec.transform(x_cat)
# print(len(vec_x_cat))

X = np.hstack((x_num, vec_x_cat))
print(len(X))
X_bigtest = np.hstack((x_num_bigtest, vec_x_cat_bigtest))
y = y_num
# y_bigtest = bigtest_set[['lcoe']].as_matrix()
y_bigtest1 = y_num_bigtest
y_bigtest = [num[0] for num in y_bigtest1]
y = [num[0] for num in y]
# var_y = stdev(y_bigtest)
# mean_y = mean(y_bigtest)
# # print(mean_y, var_y)
# y = [(num - mean_y) / var_y for num in y]
# y_bigtest = [(num - mean_y) / var_y for num in y_bigtest]

print(X[12], y[12])
accuracies = []


def run():

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # # Huang recommeds 1st hidden layer with 19 units, 2nd hidden layer with 4 units. Based on 42 inputs and 1 output.
    nn = MLPRegressor(
        hidden_layer_sizes=(50, 50), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=False,
        random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # nn = LogisticRegression(n_jobs=-1, solver='liblinear')
    # nn = LinearRegression(n_jobs=-1)

    # nn = MLPClassifier(hidden_layer_sizes=(400, 400, 400, 400, 100), tol=0.0001, solver='adam', activation='relu', random_state=54)

    # nn = RandomForestClassifier()
    # nn = KNeighborsClassifier()
    # nn = SGDClassifier()
    # nn = LinearSVR()
    # nn = SVR(kernel='rbf')
    # nn = RidgeClassifier()
    # nn = GaussianNB()
    # print(len(X_train), len(y_train))
    nn.fit(X_train, y_train)
    # # print(nn.coefs_)
    #
    # with open("regressor_coords3x3_lcoe.pickle", 'wb') as out:
    #     pickle.dump(nn, out)

    # nn = pickle.load(open('regressor_coords3x3_lcoe.pickle', 'rb'))
    # print(len(X_test), len(y_test))
    # print(X_test, y_test)
    accuracy = nn.score(X_test, y_test)
    print('small accuracy:', accuracy)
    print(X_bigtest[12].reshape(1, - 1), y_bigtest[12], nn.predict(X_bigtest[12].reshape(1, -1))[0])
    with open("comparison.dat", 'w') as comp:
        for i in range(len(y_bigtest)):
            comp.write('{} {}\n'.format(y_bigtest[i], nn.predict(X_bigtest[i].reshape(1, -1))[0]))
    print('big accuracy:', nn.score(X_bigtest, y_bigtest))
    accuracies.append(accuracy)


for _ in range(1):
    run()

print()
print('mean accuracy is:', mean(accuracies))
