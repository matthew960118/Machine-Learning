import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font
import numpy as np

data = pd.read_csv("./01/Salary_Data.csv")
#print (data)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

x= data["YearsExperience"]
y= data["Salary"]

# plt.scatter(x, y, marker="x", color="red", label="真實數據")
# plt.title("年資-薪水")
# plt.xlabel("年資")
# plt.ylabel("薪水(千)")

def plot_pred(w, b):
  y_pred = w*x+b
  plt.plot(x, y_pred, label="預測縣")
  plt.legend()

def cost_function(x, y, w, b):
  y_pred = w*x+b
  cost = (y-y_pred)**2
  return cost.sum()/len(x)

w=10
b=0
#plot_pred(w,b)

costs = np.zeros((201,201))

ws = np.arange(-100,101)
bs = np.arange(-100,101)

i=0
for w in ws:
  j=0
  for b in bs:
    costs[i,j] = cost_function(x, y, w, b)
    j=j+1
  i=i+1

ax = plt.axes(projection= "3d")
ax.xaxis.set_pane_color((0,0,0))
ax.yaxis.set_pane_color((0,0,0))
ax.zaxis.set_pane_color((0,0,0))

b_grid, w_grid = np.meshgrid(bs,ws)
ax.plot_surface(w_grid,b_grid,costs, cmap="Spectral_r")
plt.show()