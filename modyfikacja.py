import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas
# ustawienie wartości początkowych
liczofiar = 10
liczdrapiez = 5
y0 = [liczofiar, liczdrapiez]  # [ofiary,drapiezniki] w tys. warunki początkowe

#ustawienie punktów czasowych
t = np.linspace(0, 110, num=1000)

#ustawienie pararmetrów
alpha = 1.1  # współczynnik
beta = 0.4
delta = 1
gamma = 0.4
K = 100  # model z ograniczoną pojemnością środowiska dla ofiar


params = [alpha, beta, delta, gamma, K]  # ułożenie kolejności parametrów

def sim(variables, t, params):

    x = variables[0]  # poziom populacji ofiar
    y = variables[1]  # poziom populacji drapieżników


    alpha = params[0]
    beta = params[1]
    delta = params[2]
    gamma = params[3]
    K = params[4]

    # zapisanie równań
    dxdt=alpha*x*(1-(x/K))-beta*x*y
    dydt=delta * x*y-gamma *y

    return([dxdt,dydt])


yode = odeint(sim, y0, t, args=(params,))


t = t.reshape(1, 1000)
wszystko = np.concatenate((yode, t.T), axis=1)
wszystko = pandas.DataFrame(wszystko, columns=['ofiary', 'drapiezniki', 'czas'])
maxo = max(wszystko['ofiary'])
maxd = max(wszystko['drapiezniki'])
idxmaxo = wszystko['ofiary'].idxmax()
maxot = wszystko.loc[idxmaxo, 'czas']
idxmaxd = wszystko['drapiezniki'].idxmax()
maxdt = wszystko.loc[idxmaxd, 'czas']
roznicat = abs(maxot-maxdt)
mino = min(wszystko['ofiary'])
mind = min(wszystko['drapiezniki'])
print('Maksymalna liczba ofiar:', maxo)
print('Minimalna liczba ofiar:', mino)
print('Maksymalna liczba drapieżników:', maxd)
print('Minimalna liczba drapieżników:', mind)
print('Czas odnowienia populacji', roznicat)
maxt = max(wszystko['czas'])
# czas odnawiania populacji
plt.plot(wszystko['czas'], wszystko['ofiary'], color="#6CABCD")
plt.plot(wszystko['czas'], wszystko['drapiezniki'], color="#B2CD6C")
plt.plot(maxot, maxo, "o", color="#41748F"),
plt.plot(maxdt, maxd, "o", color="#727F52")
# wyswietlenie wyników
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Czas')
plt.ylabel('Liczebność populacji [tys.]')
plt.legend(['Ofiary', 'Drapieżniki'], loc='upper right')
plt.title('delta='+str(delta))
plt.show()
# przebieg czasowy
plt.plot(wszystko['czas'], wszystko['ofiary'], color="#6CABCD")
plt.plot(wszystko['czas'], wszystko['drapiezniki'], color="#B2CD6C")
# wyswietlenie wyników
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Czas')
plt.ylabel('Liczebność populacji [tys.]')
plt.legend(['Ofiary', 'Drapieżniki'], loc='upper right')
plt.title('delta='+str(delta))
plt.show()
# portret fazowy
N1 = 0
P1 = 0
N2 = K
P2 = 0
N3 = gamma/delta
P3 = (alpha/beta) * (1 - (gamma/(delta * K)))
# plt.plot(N1, P1, 'o', color='#F04BC5')
# plt.plot(N2, P2, 'o', color='#A0F04B')
plt.plot(N3, P3, 'o', color='#FFC300')
plt.plot(yode[:, 0], yode[:, 1], color="#6CABCD")
plt.xlabel('Drapieżniki')
plt.ylabel('Ofiary')
plt.title('delta='+str(delta))
# plt.legend(['Punkt stacjonarny niestabilny', 'Punkt stacjonarny stabilny'], loc='upper right')
plt.show()

