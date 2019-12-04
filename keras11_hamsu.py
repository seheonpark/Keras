# 1.데이터
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input

x = np.array(range(1, 101))
y = np.array(range(1, 101))
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=33, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=33, shuffle=False)

# 2.모델구성 순차, 함수형
# model = Sequential() 순차적 모델
input1 = Input(shape=(1, ))
xx = Dense(5, activation='relu')(input1)
xx = Dense(1000)(xx)
xx = Dense(5)(xx)
xx = Dense(1000)(xx)
xx = Dense(5)(xx)
xx = Dense(1000)(xx)
xx = Dense(5)(xx)
xx = Dense(1000)(xx)
xx = Dense(5)(xx)
xx = Dense(1000)(xx)
output1 = Dense(1)(xx)

model = Model(inputs=input1, outputs=output1)
model.summary()


# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # metrics=['accuracy']
# # acc :  1.0 loss :  1.6951153725131007e-07 
# # acc :  1.0916937576155306e-08 loss :  1.0916937576155306e-08
# model.fit(x_train, y_train, epochs=507, batch_size=1, validation_data=(x_val, y_val))

# # 4.평가 예측
# loss, mse = model.evaluate(x_test, y_test, batch_size=1)
# print("mse : ", mse)
# print("loss : ", loss)

# y_predict = model.predict(x_test)
# print(y_predict)

# # RMSE 구하기 RMSE :  7.038478273431794e-05
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기 R2 :  0.999999999399513
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print("R2 : ", r2_y_predict)