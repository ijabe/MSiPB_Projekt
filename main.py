import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# ryby - niedzwiedzie
#ustawienie wartości początkowych
y0=[10,1] # [ofiary,drapiezniki] w tys.

#ustawienie punktów czasowych
t=np.linspace(0,50,num=1000)

#ustawienie pararmetrów
alpha= 1.1 #wspólczynnik przyrostu ofiar
beta=0.4 #częstośc umierania na skutek drapieżnictwa
delta= 0.1 #wspólcynnik przyrostu ofiar
gamma=0.4 #wspólczynnik umierania drapieżników


params = [alpha, beta, delta, gamma] #ułożenie kolejności parametrów

def sim(variables, t, params):

    x=variables[0] #poziom populacji ofiar
    y=variables[1] #poziom populacji drapieznikow


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

#stabilnośc modelu przeanalizować
#portret fazowy sie zamknie
#przeanalizowac dzialanie odelu dla zmieniajacych sie wspolczynnikow alfa, gamma, beta...
#Model drapieżnik-ofiara - Model volterra lotki z modyfikacją....
#zaimplemetowac roznice miedzy maximum (czas odradzania sie miedzy ofiarami)
#wyroznic maksima i stworzyc wektor wyników
#gui?