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



def compute_gradient(x,y,w,b):
  w_gradient = (x*(w*x+b-y)).mean()
  b_gradient = ((w*x+b-y)).mean()
  return w_gradient, b_gradient

def cost_function(x, y, w, b):
  y_pred = w*x+b
  cost = (y-y_pred)**2
  return cost.sum()/len(x)

def gradient_descent(x,y,w_init,b_init,learning_rate,cost_function, gradient_function ,runtime, p_iter=1000):
  
  cost_hist=[]
  w_hist=[]
  b_hist=[]
  w=w_init
  b=b_init
  
  for i in range(runtime):
    w_gradient, b_gradient = gradient_function(x,y,w,b)
    w = w - w_gradient*learning_rate
    b = b - b_gradient*learning_rate
    cost = cost_function(x,y,w,b)

    w_hist.append(w)
    b_hist.append(b)
    cost_hist.append(cost)
    if i%p_iter==0:
      print(f"Ieration {i:5} : Cost{cost:.2e}, w:{w:.2e}, b:{b:.2e}, w_gradient:{w_gradient:.2e}, b_gradient:{b_gradient:.2e}")
  return w, b, w_hist, b_hist, cost_hist
  
w_init = 0
b_init = 0
learning_rate = 0.01

w_final, b_final, w_hist, b_hist, cost_hist=gradient_descent(x,y,w_init,b_init,learning_rate,cost_function,compute_gradient ,20000)

data_limit = 100

plt.plot(np.arange(0,data_limit),cost_hist[:data_limit])
plt.show()