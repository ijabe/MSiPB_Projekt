import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#ustawienie wartości początkowych
y0=[10,1] # [ryby,niedzwiedzie] w tys.

#ustawienie punktów czasowych
t=np.linspace(0,50,num=1000)

#ustawienie pararmetrów
alpha= 1.1
beta=0.4
delta= 0.1
gamma=0.4


params = [alpha, beta, delta, gamma] #ułożenie kolejności parametrów

def sim(variables, t, params):

    x=variables[0] #poziom populacji ryb
    y=variables[1] #poziom populacji niedźwiedzi


    alpha = params[0]
    beta = params[1]
    delta = params[2]
    gamma = params[3]

    #zapisanie równań
    dxdt=alpha*x-beta*x*y
    dydt=delta * x*y-gamma *y

    return([dxdt,dydt])

y=odeint(sim, y0, t, args=(params,))

f,(ax1,ax2)=plt.subplots(2)

line1, = ax1.plot(t,y[:,0], color="b")
line2, = ax2.plot(t,y[:,1], color="r")

ax1.set_ylabel("Ryby (w tys.)")
ax2.set_ylabel("Niedźwiedzie (w tys.)")
ax2.set_xlabel("Czas")

#wyswietlenie wyników
plt.show()