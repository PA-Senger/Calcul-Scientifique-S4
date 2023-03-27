"""TP3 Integration numerique."""

import numpy as np
import matplotlib.pyplot as plt

# Choix des bornes de l'intervalle
a = 0
b = np.pi / 2
# b = 1  # borne sup question bonus

# Definition de la fonction a integrer


def f(x):
    """f(x) = sin(x).

    param x: float, angle
    return: float, sin(x)
    """
    return np.sin(x)


def f1(x):
    """f1(x) = 2, fonction constante.

    param x : float, angle
    return : float
    """
    return 2


def f2(x):
    """f2(x) = 2x + 1.

    param x : float, angle
    return : float
    """
    return 2 * x + 1


def f3(x):
    """f3(x) = x**2 + x + 1.

    param x : float, angle
    return : float
    """
    return x**2 + x + 1


# Definition de la primitive de la fonction a integrer
# afin de tester les methodes d'integration
def F(x):
    """Primitive de f(x) = sin(x).

    param x: float, angle
    return: float, F(x) = -cos(x).
    """
    return - np.cos(x)


def F1(x):
    """Primitive de f1(x) = 2.

    param x : float, angle
    return : float, F(x) = 2x.
    """
    return 2 * x


def F2(x):
    """Primitive de f2(x) = 2x + 1.

    param x : float, angle
    return : float.
    """
    return x**2 + x


def F3(x):
    """Primitive de f3.

    param x : float, angle
    return : float.
    """
    return (x**3 / 3) + (x**2 / 2) + x


# Calcul de la valeur exacte de l'integrale I de f
I = F(b) - F(a)
#I = F1(b) - F1(a)
#I = F2(b) - F2(a)
#I = F3(b) - F3(a)

# Integration numerique d'une fonction f en n rectangles
# sur le segment [a, b].

# Methode des rectangles a gauche.


def int_recg(f, a, b, n):
    """Integration rectangle gauche.

    param f : fonction mathematique
    param a, b : float, bornes d'Integration
    param n : int, nombre de rectangles utilises pour l'approximation de l'integrale.
    return : float, approximation de l'integral entre a et b.
    """
    h = (b - a) / n  # h, pas d'espace -> subdivion reguliere
    x = a  # x, sommet du rectangle, point de gauche
    S = 0  # S, la surface
    for i in range(n):
        S += f(x) * h  # somme des surfaces, base h * hauteur f(x)
        x += h   # et on incremente x
    return S


# Methode des rectangles a droite.
def int_recd(f, a, b, n):
    """Integration rectangle droite.

    param f : fonction mathematique
    param a, b : float, bornes d'Integration
    param n : int, nombre de rectangles utilises pour approximer l'intergrale
    return : float, approximation de l'integral entre a et b.
    """
    h = (b - a) / n  # h, pas d'espace
    x = a + h  # x, sommet du rectangle, point de droite
    S = 0  # S, la surface
    for i in range(n):
        S += f(x) * h  # base h * hauteur f(x)
        x += h  # on incremente x
    return S


# Methode des rectangles au milieu.
def int_recm(f, a, b, n):
    """Integration rectangle milieu.

    param f : fonction mathematique
    param a, b : float, bornes d'Integration
    param n : int, nombre de rectangles utilises pour approximer l'intergrale
    return : float, approximation de l'integral entre a et b.
    """
    h = (b - a) / n  # h, pas d'espace
    x = a + h / 2  # x, sommet du rectangle
    S = 0  # S, la surface
    for i in range(n):
        S += f(x) * h  # base h * hauteur f(x)
        x += h  # on incremente x
    # print(S)
    return S


# Methode des trapeze.
def int_trap(f, a, b, n):
    """Integration trapeze.

    param f : fonction mathematique
    param a, b : float, bornes d'Integration
    param n : int, nombre de rectangles utilises pour approximer l'intergrale
    return : float, approximation de l'integral entre a et b.
    """
    h = (b - a) / n  # h, pas d'espace
    x = a  # x, sommet
    S = 0  # S, la surface
    for i in range(n):
        S += (h / 2) * (f(x) + f(x + h))  # (b-a)/2 * f(a)+f(b)
        x += h
    return S


# Methode de Simpson.
def int_sim(f, a, b, n):
    """Integration Simpson.

    param f : fonction mathematique
    param a, b : float, bornes d'Integration
    param n : int, nombre de pas dans la subdivision
    return : float, approximation de l'integral entre a et b.
    """
    h = (b - a) / n   # h, pas d'espace
    x = a  # x, sommet
    S = 0  # S, la surface
    for i in range(n):
        S += (h / 6) * (f(x) + 4 * (f((2 * x + h) / 2)) + f(x + h))  # formule de Simpson
        x += h
    return S


# Calcul de la valeur de l'integrale approchee S_rg de f
# par la methode des rectangles a gauche pour n = 100, 1000 et 10000
# et de l'erreur en valeur absolue E_rg avec l'integrale exacte I

# a gauche
n100 = 100  # nombre de rectangles
S_rg_100 = int_recg(f, a, b, n100)  # Surface rectangle Gauche
E_rg_100 = np.abs(I - S_rg_100)  # Erreur rectangle Gauche, E = |I - S|
print("Fonction : f(x)=sin(x) sur l'intervalle [" + repr(a) + "," + repr(b) + "]")
print('Integrale exacte : \nI = ' + repr(I))
print("")
print("Integration par la methode des rectangles a gauche :")
print('Integrale approchee et erreur pour :')
print('n = ' + repr(n100) + '   ; S = ' + repr(S_rg_100) + ' ; ' + 'E = ' + repr(E_rg_100))
n1000 = 1000  # nombre de rectangles
S_rg_1000 = int_recg(f, a, b, n1000)  # Surface rectangle Gauche
E_rg_1000 = np.abs(I - S_rg_1000)  # Erreur rectangle Gauche
print('n = ' + repr(n1000) + '   ; S = ' + repr(S_rg_1000) + ' ; ' + 'E = ' + repr(E_rg_1000))
n10000 = 10000  # nombre de rectangles
S_rg_10000 = int_recg(f, a, b, n10000)  # Surface rectangle Gauche
E_rg_10000 = np.abs(I - S_rg_10000)  # Erreur rectangle Gauche
print('n = ' + repr(n10000) + '   ; S = ' + repr(S_rg_10000) + ' ; ' + 'E = ' + repr(E_rg_10000))
print("")

# a droite
n100 = 100  # nombre de rectangles
S = int_recd(f, a, b, n100)  # Surface rectangle droite
E = np.abs(I - S)  # Erreur rectangle droite, E = |I - S|
print("Integration par la methode des rectangles a droite :")
print('Integrale approchee et erreur pour :')
print('n = ' + repr(n100) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n1000 = 1000  # nombre de rectangles
S = int_recd(f, a, b, n1000)  # Surface rectangle droite
E = np.abs(I - S)  # Erreur rectangle droite
print('n = ' + repr(n1000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n10000 = 10000  # nombre de rectangles
S = int_recd(f, a, b, n10000)  # Surface rectangle droite
E = np.abs(I - S)  # Erreur rectangle droite
print('n = ' + repr(n10000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
print("")

# au milieu
n100 = 100  # nombre de rectangles
S = int_recm(f, a, b, n100)  # Surface rectangle milieu
E = np.abs(I - S)  # Erreur rectangle milieu, E = |I - S|
print("Integration par la methode des rectangles du point milieu :")
print('Integrale approchee et erreur pour :')
print('n = ' + repr(n100) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n1000 = 1000  # nombre de rectangles
S = int_recm(f, a, b, n1000)  # Surface rectangle milieu
E = np.abs(I - S)  # Erreur rectangle milieu
print('n = ' + repr(n1000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n10000 = 10000  # nombre de rectangles
S = int_recm(f, a, b, n10000)  # Surface rectangle milieu
E = np.abs(I - S)  # Erreur rectangle milieu
print('n = ' + repr(n10000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
print("")

# tapezes
n100 = 100  # nombre de trapezes
S = int_trap(f, a, b, n100)  # Surface trapeze
E = np.abs(I - S)  # Erreur, E = |I - S|
print("Integration par la methode des trapezes :")
print('Integrale approchee et erreur pour :')
print('n = ' + repr(n100) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n1000 = 1000
S = int_trap(f, a, b, n1000)
E = np.abs(I - S)
print('n = ' + repr(n1000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n10000 = 10000
S = int_trap(f, a, b, n10000)
E = np.abs(I - S)
print('n = ' + repr(n10000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
print("")

# Simpson
n100 = 100  # nombre de pas
S = int_sim(f, a, b, n100)  # Surface trapeze
E = np.abs(I - S)  # Erreur, E = |I - S|
print("Integration par la methode de Simpson :")
print('Integrale approchee et erreur pour :')
print('n = ' + repr(n100) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n1000 = 1000
S = int_sim(f, a, b, n1000)
E = np.abs(I - S)
print('n = ' + repr(n1000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))
n10000 = 10000
S = int_sim(f, a, b, n10000)
E = np.abs(I - S)
print('n = ' + repr(n10000) + '   ; S = ' + repr(S) + ' ; ' + 'E = ' + repr(E))

# Representations :
# de la valeur exacte I de l'intergrale de f entre 0 et pi/2 ,
# des integrales approchee S par la methode des rectangles
# a gauche/droite/milieu, des trapezes et de Simpson.
# Ainsi que de l'erreur E en valeur absolue avec l'integrale exacte I en
# echelle standard puis en echelle loglog pour differentes valeurs de nmin et
# nmax par increments de step_n
nmin = 500
nmax = 20000
step_n = 100
n_list = np.arange(nmin, nmax, step_n)
S_rg_list = np.asarray([int_recg(f, a, b, n) for n in n_list])
S_rd_list = np.asarray([int_recd(f, a, b, n) for n in n_list])
S_rm_list = np.asarray([int_recm(f, a, b, n) for n in n_list])
S_trap_list = np.asarray([int_trap(f, a, b, n) for n in n_list])
S_sim_list = np.asarray([int_sim(f, a, b, n) for n in n_list])
E_rg_list = np.abs(I - S_rg_list)
E_rd_list = np.abs(I - S_rd_list)
E_rm_list = np.abs(I - S_rm_list)
E_trap_list = np.abs(I - S_trap_list)
E_sim_list = np.abs(I - S_sim_list)
# Representation de I (horizontale) et de S en fonction de n
plt.plot(n_list, S_rg_list, label='Methode des rectangles a gauche')
plt.plot(n_list, S_rd_list, label='Methode des rectangles a droite')
plt.plot(n_list, S_rm_list, label='Methode des rectangles point milieu')
plt.plot(n_list, S_trap_list, label='Methode des trapezes')
plt.plot(n_list, S_sim_list, label='Methode de Simpson')
plt.axhline(y=I, color='red', linestyle='dashed', label='Valeur exacte')
plt.title("Integration numerique de la fonction sinus(x) entre 0 et pi/2")
plt.xlabel('n = nombre de pas dans la subdivision')
plt.ylabel("S = valeur de l'integrale")
plt.legend()
plt.show()
# Representation de l'erreur E en fonction de n
plt.plot(n_list, E_rg_list, label='Methode des rectangles a gauche')
plt.plot(n_list, E_rd_list, label='Methode des rectangles a droite')
plt.plot(n_list, E_rm_list, label='Methode des rectangles point milieu')
plt.plot(n_list, E_trap_list, label='Methode des trapezes')
plt.plot(n_list, E_sim_list, label='Methode de Simpson')
plt.title("Evolution de l'erreur d'integration en fonction de n")
plt.xlabel('n = nombre de pas dans la subdivision')
plt.ylabel("E = erreur de l'integration numerique")
plt.legend()
plt.show()
# Representation de l'erreur E en fonction de n en echelle loglog
plt.loglog(n_list, E_rg_list, label='Methode des rectangles a gauche')
plt.loglog(n_list, E_rd_list, label='Methode des rectangles a droite')
plt.loglog(n_list, E_rm_list, label='Methode des rectangles point milieu')
plt.loglog(n_list, E_trap_list, label='Methode des trapezes')
plt.loglog(n_list, E_sim_list, label='Methode de Simpson')
plt.title("Erreur en fonction de n dans un repere log-log")
plt.xlabel('n = nombre de pas dans la subdivision')
plt.ylabel("E = erreur d'integration")
plt.grid(True)
plt.legend()
plt.show()
