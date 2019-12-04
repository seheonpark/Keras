# 1.데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(1, 101))
print(x)
# x_train = x[:60]
# y_train = y[:60]
# x_test = x[:80]
# y_test = y[:80]
# x_val = x[60:80]
# y_val = y[60:80]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=33, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=33, shuffle=False)
# 2.모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(501, input_dim=1, activation='relu'))
model.add(Dense(5, input_shape=(1, ), activation='relu')) # R2 :  0.9999999999996362 R2 :  0.999999999999537 R2 :  0.9999999999998347
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(1))

#model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse']) # metrics=['accuracy']
# acc :  1.0 loss :  1.6951153725131007e-07 
# acc :  1.0916937576155306e-08 loss :  1.0916937576155306e-08
model.fit(x_train, y_train, epochs=507, batch_size=1, validation_data=(x_val, y_val))

# 4.평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기 RMSE :  7.038478273431794e-05
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기 R2 :  0.999999999399513
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)