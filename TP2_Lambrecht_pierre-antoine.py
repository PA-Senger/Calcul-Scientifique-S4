""""TP2. Interpolation de Lagrange."""

import numpy as np
import pylab as pl


# 1
def f(x):
    """Fonction Sinus(x).

    param x : float
    return : float, Sin(x)
    """
    return np.sin(x)


# 2, on definies les bornes de notre intervalle d'interpolation
xmin = -3 * np.pi
xmax = 3 * np.pi

# 3
nb_pts_interpol = 9  # fixee, a varier pour augmenter la precision de l'approximation de la fonction
x_inter = np.linspace(xmin, xmax, nb_pts_interpol)  # liste linspace(borne inf, borne sup, nb de points de la subdivion reguliere)
#print("x_inter=", x_inter)

# 4
y_inter = f(x_inter)  # liste sin(x_inter)
#print("y_inter =", y_inter)


# 5
def poly_lagrange(x0, k, points):
    """Renvoi la valeur en un point x du k-eme poly de Lagrange L_k.

    param x0 : float
    param k : float
    param points : list, xi
    return : float
    """
    L = 1
    for i in range(len(points)):  # par iteration on calcul de i-eme ou plutot ici le k-eme polynome de Lagrange
        if i != k:            # tant qu'on a pas l'indice L_k que l'on souhaite
            L = L * ((x0 - points[i]) / (points[k] - points[i]))  # formule polynome de Lagrange
    return L


# 6
def poly_interpolation(x0, xi, yi):
    """Renvoi la valeur en un point x0 du polynôme d'interpolation de Lagrange P
    construit a partir des points d'interpolation X=(xi, yi).

    param x : float
    param xi : list, abscisses des points d'interplations
    param yi : list, ordonnees des points d'interpolations
    return : float, P(x0)
    """
    P = 0  # on initialise a 0
    for i in range(len(xi)):  # len(xi) = len(yi)
        P += yi[i] * poly_lagrange(x0, i, xi)  # P(x)=(somme de i=0 a n) y_i * L_i
    return P


# 8
N_plot = 200

# 9
X = np.linspace(xmin, xmax, N_plot)  #  liste, subdivision reguliere de l'intervalle [xmin, xmax] en N_plot points

# 10
Y = f(X)  # liste, images de X par sin(x)

# 11
P = []  # initialistation liste vide, va contenir valeurs du polynome d'interpolation de Lagrange
for i in range(len(X)):  # boucle for qui rempli la liste P
    P += [poly_interpolation(X[i], x_inter, y_inter)]


# 12
# le nb de pts d'interpolation est a la 22 eme ligne
def trace1():
    """Fonction non formel juste pour les trace de pylab.
    Contient des variables global de ci dessus"""
    pl.figure(figsize=(18, 10))  # image et taille.
    pl.title("Graphes", fontsize=22)  # titre au dessus du graphe et taille
    pl.xlabel("x", fontsize=16)  # nom abscisses et taille
    pl.ylabel("y", fontsize=16)  # nom ordonnee et taille
    pl.plot(x_inter, y_inter, 'ro', label="Points d'interpolations")  # (a)
    pl.plot(X, f(X), 'g-', label="Sinus(x)")                        # (b)
    pl.plot(X, P, 'b-', label="P(x)")                               # (c)
    pl.legend()                         # affiche les labels sur l'image
    pl.grid(True)       # ajoute une grille en fond
    pl.show()           # affiche les plot apelles


trace1()  # appel trace1() et creer notre image


# 14
# j'ai ecrit la procedure sous forme d'une fonction en essayant d'eviter
# d'avoir des variables globales
def error(nb_pts_erreur, nb_pts_interpol, f, xmin, xmax):
    """Evalue l'erreur en norme l_infinie entre P_n et la fonction f.

    param nb_pts_erreur : int, nb de pts dans la subdivision
    param nb_pts_interpol : int
    param f : Fonction
    param xmin, xmax : float, intervalle
    return : couple of float (erreur, x): erreur = sup{f - P} en x = ...
    """
    x = None  # abscisse du sup
    x_inter = np.linspace(xmin, xmax, nb_pts_interpol)  # subdivision
    y_inter = f(x_inter)  # image de la subdivion par la fonction f(x) = sin(x)
    x_erreur = np.linspace(xmin, xmax, nb_pts_erreur)  # subdivision reguliere
    Poly = poly_interpolation(x_erreur[0], x_inter, y_inter)  # polynome d'interpolation de depart
    erreur = abs(f(x_erreur[0]) - Poly)  # |f(x0) - P(x0)|
    for i in range(nb_pts_erreur):  # on teste nb_pts_erreur points
        P = poly_interpolation(x_erreur[i], x_inter, y_inter)  # poly de Lagrange evalue en x_erreur[i]
        diff = abs(f(x_erreur[i]) - P)  # |f(xi) - P(xi)|
        if diff > erreur:
            erreur = diff     # on remplace car on veur le Sup
            x = x_erreur[i]  # abscisse du sup
    return erreur, x


# Test de la fonction error() (egalement utile pour la question 15)

nb_pts_erreur = 2000  # nombre de pts tester
nb_pts_interpol = 0
for i in range(25):
    nb_pts_interpol += 1
    print("Erreur = {} en x = {} pour {} points d'interpolation de P(x) et {} points d'erreurs"
          .format(error(nb_pts_erreur, nb_pts_interpol, f, xmin, xmax)[0],
          error(nb_pts_erreur, nb_pts_interpol, f, xmin, xmax)[1], nb_pts_interpol, nb_pts_erreur))


# 15
def g(nb_pts_erreur, f, n):
    """Erreur en fonction de n.

    param nb_pts_erreur : int
    param f : fonction mathematique
    param n : int, jusqu'a ou on veut aller
    return : couple (N, E); N : list n_i eme pts d'interpol ; E : list erreur
    """
    N = []  # initialistation liste vide
    E = []  # initialistation liste vide
    nb_pts_interpol = 1  # initialistation a 1 point d'interpolation
    while nb_pts_interpol <= n:  # de 1 a n
        nb_pts_interpol += 1  # on incremente le nb de pts d'interpolations
        N += [nb_pts_interpol]  # le "n" pts d 'interpolations correspondant a l'erreur
        E += [error(nb_pts_erreur, nb_pts_interpol, f, xmin, xmax)[0]]  # on enregistre la valeur de erreur dans notre liste
    return N, E


def trace2():
    """Graphe de l'erreur en fonction de n."""
    N, E = g(1000, f, 30)  # stockage du return de la fonction g dans des variables
    pl.figure(figsize=(18, 10))  # image pylab
    pl.title("Evolution de l'erreur en fonction de n", fontsize=22)  # titre
    pl.xlabel("n", fontsize=16)  # abscisses
    pl.ylabel("Erreur", fontsize=16)  # ordonnee
    pl.plot(N, E, 'rx', label="Sup|Sin(x) - P(x)|")  # notre plot
    pl.legend()  # legend du label
    pl.grid(True)  # ajout de la grille de fond
    pl.show()      # affichage graphique


trace2()  # appel trace2() et creer notre image
# l'image prend un peu de temps a ce creer
