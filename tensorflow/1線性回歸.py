import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./randomdata.csv")

x = data["x"]
y = data["y"]
w = 1
b = 0
learning_rate = 0.0001

fig, ax = plt.subplots()
plt.ion()

def loss(x, y, w, b):
    loss = 0
    for i in range(len(x)):
        loss += (w * x[i] + b - y[i]) ** 2
    return loss

def compute_gradient(x, y, w, b):
    w_gradient = 0
    b_gradient = 0
    for i in range(len(x)):
        w_gradient += x[i] * (x[i] * w + b - y[i])
        b_gradient += (x[i] * w + b - y[i])
    return w_gradient, b_gradient

plt.scatter(x, y)
line, = ax.plot([-5, 30], [-5*w + b, 30 * w + b])

for i in range(2000):
    print(f"w:{w:10.2e}, b:{b:10.2e}, loss:{loss(x, y, w, b):10.2e}")
    w_gradient, b_gradient = compute_gradient(x, y, w, b)
    w -= learning_rate * w_gradient
    b -= learning_rate * b_gradient

    line.set_ydata([-5*w + b, 30 * w + b])
    plt.draw()
    plt.pause(0.01)

plt.ioff()  
plt.show()
