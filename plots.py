'''
import numpy as np
import matplotlib.pyplot as plt

# Create data
N = 15
g1 = (2 + np.random.rand(N), 2 + 3*np.random.rand(N))
g2 = (2 + np.random.rand(N), 5 + 1.5*np.random.rand(N))
g3 = (4 + 0.9*np.random.rand(N), 2 + 1.5*np.random.rand(N))
g4 = (4+ 0.9 * np.random.rand(N), 4 + 3*np.random.rand(N))
g5 = (np.linspace(1,5,500), np.full(500,3))
g6 = (np.full(500,3.5),np.linspace(0,7,500))
#g7 = (np.full(500,2),np.linspace(7,10,500))




data = (g1, g2, g3, g4, g5)
colors = ("red", "green", "blue", 'orange','black')
groups = ("class 1", "class 2", "class 3",'class 4' ,None)

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

for data, color, group in zip(data, colors, groups):
    x, y = data
    if group is None:
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=5)
    else:
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=50)

ax.title.set_text('Horizontal split')

ax1 = fig.add_subplot(1, 2, 2)
data = (g1, g2, g3, g4, g6)


for data, color, group in zip(data, colors, groups):
    x, y = data
    if group is None:
        ax1.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=5)
    else:
        ax1.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=50)

ax1.title.set_text('Vertical split')

ax.set_xlim([1.5,5])
ax.set_ylim([1.5,7])
ax1.set_xlim([1.5,5])
ax1.set_ylim([1.5,7])
#plt.show()

'''

import math


print(-math.log(0.25) - (2/3)*(-(15/20)*math.log(15/40) -(7/40)*math.log(7/40) - (3/40)*math.log(3/40)) - (1/3)*(-(8/20)*math.log(8/20) - (12/20)*math.log(12/20)))

print(3/4 - (2/3)*(1-2*(15/40)*(15/40) - (7/40)*(7/40) - (3/40)*(3/40)) - (1/3)*(1-(8/20)*(8/20) - (12/20)*(12/20)))

print(3/4 - (2/3)*(25/40) - (1/3)*(8/20))

print(-math.log(0.25) + math.log(0.5))

print(3/4 - (1 - 0.25 - 0.25))

print(1 - 0.25 - 0.25 - 0.25)



