import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font
import numpy as np
from matplotlib.widgets import RangeSlider


p=[]
with open("./primes-to-100k.txt") as f:
  p = list(map(int,f.read().split("\n")))

axfreq = plt.axes([0.1, 0.1, 0.8, 0.05])
slider = RangeSlider(
    ax=axfreq,
    label='Offset',
    valmin=-5,
    valmax=5,
    valinit=0,
)
plt.subplots_adjust(left=0.1, bottom=0.25)


index=10
y=p[0:index]
x=range(1,len(p[0:index])+1)
plt.plot(x,y, color="red")
plt.show()