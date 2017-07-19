import pickle
import numpy as np
import pandas as pd

# pickles MLPRegressor by Sciki Learn after training.
nn = pickle.load(open('regressor_coords3x3_lcoe.pickle', 'rb'))


def nn_predict(x):
    x = np.reshape(x, (1, -1))
    return nn.predict(x)[0]

if __name__ == '__main__':

    # df = pickle.load(open('pickle_full_data.pickle', 'rb'))
    #
    # df['A'] = df['A'].apply(str)
    # df['B'] = df['B'].apply(str)
    # df['C'] = df['C'].apply(str)
    # df['D'] = df['D'].apply(str)
    # df['E'] = df['E'].apply(str)
    # df['F'] = df['F'].apply(str)
    # df['G'] = df['G'].apply(str)
    # df['H'] = df['H'].apply(str)
    # df['I'] = df['I'].apply(str)
    # df['J'] = df['J'].apply(str)
    #
    # bigtest_set = df.sample(frac=0.01, replace=True)
    # numeric_cols = ['nbins', 'dir_real', 'dir_artif']
    # x_num_bigtest = bigtest_set[numeric_cols].as_matrix()
    # max_x = np.amax(x_num_bigtest, 0)
    # x_num_bigtest = x_num_bigtest / max_x
    # cat_bigtest = bigtest_set.drop(numeric_cols + ['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1)
    # x_cat_bigtest = cat_bigtest.to_dict(orient='records')
    # vec = pickle.load(open('pickle_vectorizer.pickle', 'rb'))
    # vec_x_cat_bigtest = vec.transform(x_cat_bigtest)
    # X_bigtest = np.hstack((x_num_bigtest, vec_x_cat_bigtest))
    # y_bigtest = bigtest_set[['lcoe']].as_matrix()
    # print(X_bigtest[2342].tolist())
    # with open("comparison.dat", 'w') as comp:
    #     for i in range(len(y_bigtest)):
    #         # print(y_bigtest[i], nn.predict(X_bigtest[i].reshape(1, -1))[0])
    #         comp.write('{} {}\n'.format(y_bigtest[i][0], nn.predict(X_bigtest[i].reshape(1, -1))[0]))
    # print('big accuracy:', nn.score(X_bigtest, y_bigtest))

    print(nn_predict([0.28, 1.0, 0.3333333333333333, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
