import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#ustawienie wartości początkowych
y0=[10,1] # [ryby,niedzwiedzie] w tys.

#ustawienie punktów czasowych
t=np.linspace(0,110,num=1000)

#ustawienie pararmetrów
alpha= 1.1 # współczynnik
beta=0.4
delta= 0.1
gamma=0.4
K=100 # model z ograniczoną pojemnością środowiska dla ofiar


params = [alpha, beta, delta, gamma, K] #ułożenie kolejności parametrów

def sim(variables, t, params):

    x=variables[0] #poziom populacji ryb
    y=variables[1] #poziom populacji niedźwiedzi


    alpha = params[0]
    beta = params[1]
    delta = params[2]
    gamma = params[3]
    K = params[4]

    #zapisanie równań
    dxdt=alpha*x*(1-(x/K))-beta*x*y
    dydt=delta * x*y-gamma *y

    return([dxdt,dydt])

y=odeint(sim, y0, t, args=(params,))

y=odeint(sim, y0, t, args=(params,))

v = [0, 110, 0, 20]
plt.plot(t,y[:,0], color="m")
plt.plot(t,y[:,1], color="c")
plt.axis(v)
#wyswietlenie wyników
plt.show()
