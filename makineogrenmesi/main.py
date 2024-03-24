# RNN (Recurrent Neural Network - Tekrarlayan Sinir Ağı) kullanarak
# bir zaman serisini  tahmin etmek için bir model oluşturur ve eğitir.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

N = 1000
Tp = 800

t = np.arange(0, N)
x = np.sin(0.02 * t) + 2 * np.random.rand(N)
df = pd.DataFrame(x)

plt.plot(df)
plt.show()

values = df.values
train, test = values[0:Tp, :], values[Tp:N, :]

step = 4
test = np.append(test, np.repeat(test[-1, :], step))
train = np.append(train, np.repeat(train[-1, :], step))

def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)

trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(1, step), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()



