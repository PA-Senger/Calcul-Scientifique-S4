"""S4 math calcul scientifique TP1, fonctions non lineaires."""

import numpy as np


# Exercice 3

# Question 2

# (d)
def g(x, C):
    """Fonction g qui renvoi g_c(x).

    param x : float
    param C : float != 0
    """
    return C * ((2 - x) * np.exp(-x) - 1) + x


# print(g(0, 1 / 4))


# (e)
def pointfixe(f, C, x0, eps):
    """Calcule le point fixe d'une fonction f.

    param f : fonction mathematique bien definie
    param C : float, constante
    param x0 : float, point initial
    param eps : float, epsilon fixee, moralement petit
    return : couple, (point fixe, nb d'interations)
    """
    xn = x0     # changement d'indice pour une meilleur lecture
    n = 0       # initialistation
    erreur = np.abs(f(xn, C) - xn)  # erreur = |f(xn)-xn|
    while erreur > eps:  # tant que notre erreur est superieur a notre epsilon fixee
        xn = f(xn, C)  # recurrence
        erreur = np.abs(f(xn, C) - xn)  # nouvelle valeur de notre erreur
        n += 1  # incrementation du compteur n
    return xn, n


# (f)
epsilon = 1e-10
(xbar, n) = pointfixe(g, 1 / 4, 0, epsilon)
print("Methode point fixe : xbar = {} apres {} iterations pour epsilon = {}".format(xbar, n, epsilon))

# (g)
# nmax (de la question 2)c)):
# n <= (eps(1-K)/||x1-x0||) / (ln(K))
# avec eps = e(-10) et K = 1- e**(-1)/2
# on trouve n <= 114.8
# donc 115 est un majorant
# nmax = 115
# donc on a bien n <= nmax

# Question 3

# (a)


def f(x):
    """ f(x) = 2 - x - exp(x).

    param x : float
    return : float
    """
    return 2 - x - np.exp(x)


def df(x):
    """Derivee de f(x).

    param x : float
    return : float.
    """
    return -1 - np.exp(x)


# (b)(c)(d)
def newton(f, df, x, eps):
    """Methode iterative de Newton.

    param f : fonction mathematique bien definie
    param x : float
    param eps : float, epsilon
    return : couple, (xbar, n) ou n est le nombre d'iteration
    """
    n = 0       # initialistation
    erreur = 2 * eps  # valeur superieur a epsilon pour rentrer dans la boucle
    while erreur > eps:  # tant que erreur est superieur a notre epsilon fixee
        xn = x - (f(x) / df(x))  # recurrence
        erreur = np.abs(xn - x)  # nouvelle valeur d'erreur, |xn - x|
        x = xn
        n += 1  # incrementation du compteur n
    return x, n


epsilon = 1e-10
xbar, n = (newton(f, df, 0, epsilon))  # on extrait et reassigne le return de newton()
print("Methode de Newton  : xbar = {} apres {} iterations (pour epsilon = {})".format(xbar, n, epsilon))


# Sur cette exemple la differnce en precision des 2 methodes ne semble pas
# significative mais la methode de Newton est beaucoup plus rapide, ce qui est
# evidement un plus en temps de calcule et tout les avantages qui en
# decoule.
