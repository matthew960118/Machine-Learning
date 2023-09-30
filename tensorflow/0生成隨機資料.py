import pandas as pd
import matplotlib.pyplot as plt
import random

pd.set_option('display.max_rows', None)

def f(x, a=1, b=0):
    return a*x+b

randomdata = pd.DataFrame(columns=["x","y"])

for j in range(4):
  for i in range(25):
      a = f(i,a=3,b=10) + random.uniform(-10,10)
      randomdata.loc[len(randomdata)] = {"x":i,"y":a}

x = randomdata["x"]
y = randomdata["y"]

plt.scatter(x,y)
plt.show()
randomdata.to_csv("./randomdata.csv")
