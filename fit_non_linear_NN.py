from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from random import uniform

x = np.array([uniform(- 5.0, 5.0) for _ in range(1000)]).reshape(-1, 1)
# print(x)
y = x ** 2.0

nn = MLPRegressor(
    hidden_layer_sizes=(7,),  activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=0, tol=0.001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

n = nn.fit(x, y)
test_x = np.arange(- 5.0, 5.0, 0.5).reshape(-1, 1)
# print(test_x)
test_y2 = test_x ** 2.0
test_y = nn.predict(test_x)
accuracy = nn.score(test_x, test_y2)
print(accuracy, "%")
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=1, c='b', marker="s", label='real')
ax1.scatter(test_x, test_y, s=10, c='r', marker="o", label='NN Prediction')
plt.show()