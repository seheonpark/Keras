from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
x2 = np.array([11, 12, 13, 14, 15])

model = Sequential()
model.add(Dense(101, input_dim=1, activation='relu'))
for i in range(100, 0, -1):
    model.add(Dense(i))

model.summary()


# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs=100)

# loss, acc = model.evaluate(x, y)
# print("acc : ", acc)
# print("loss : ", loss)

# y_predict = model.predict(x2)
# print(y_predict)