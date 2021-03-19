import matplotlib.pyplot as plt
import numpy as np

I = 10**3
mu = 0.2
r = 0.1
sigma = 1
def f(alpha) :
    y = (mu - r)*(1-alpha*(1+r)*I)/(alpha*((mu-r)**2 + sigma**2))
    return(y)
X = np.linspace(0, 0.001, 100)

plt.plot(X, f(X))
plt.xlabel("alpha")
plt.ylabel("z")
plt.savefig("/Users/leore/Desktop/StanfordCourses/CME241/RL-book/rl/HW6/q1.png")
