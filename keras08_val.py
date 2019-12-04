# 1.데이터
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x_val = np.array([101, 102, 103, 104, 105])
y_val = np.array([111, 107, 103, 204, 1115])

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
model.fit(x_train, y_train, epochs=505, batch_size=1, validation_data=(x_val, y_val))

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