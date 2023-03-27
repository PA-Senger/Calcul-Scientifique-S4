"""TP4. Resolution numerique de systemes lineaires."""

import numpy as np

# Exercice 1 : Decomposition LU d'une matrice

# Question 2

A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]  # matrice A
# for l in A:
#    print(l)

# a) Dimension d'une matrice
n, p = np.shape(A)  # avec A un matrice de dim n*p, (n lignes, p colones)
#print("dim A = {} x {}".format(n, p))

# b) Initialisation L = I_n

L = np.identity(n)  # matrice identite de taille n*n
# print(L)

# c)

U = np.copy(A)  # copie de A dans une nouvelle variable U
# print(U)


def init_matrice(A):  # cas general pour une matrice carree quelconque
    """Renvoie une copie de la matrice A de taille n*n et la matrice identite I_n.

    param A : array, matrice carree
    return : array, U (copie de A), I (matrice identite de meme dim que A).
    """
    n, p = np.shape(A)  # dimensions de la matrice A
    if n != p:  # verif erreur utilisateur
        print("Erreur : la matrice doit etre carree")
        return None
    U = np.copy(A)  # copie de A dans une nouvelle variable U
    I = np.identity(n)  # I_n
    return U, I


# print(init_matrice(A)[0])
# print(init_matrice(A)[1])

# d) Decomposition LU
def LU(A):
    """Decomposition de A = LU.

    param A : array, matrice carree
    return : arrays, (L, U). L = triangulaire inf, U = triangulaire sup
    """
    n, p = np.shape(A)  # dim de A
    U, L = init_matrice(A)  # init U et L
    for k in range(1, n):  # de la deuxieme ligne jusqu'a la fin car la premiere ligne reste inchangee dans une matrice triangulaire sup
        pivot = U[k - 1][k - 1]  # pivot
        for i in range(k, n):  # depuis notre ligne actuelle jusqu'a la fin
            c = U[i][k - 1] / pivot  # stockage du coef du pivot avant modif de la ligne
            U[i, :] = U[i, :] - c * U[k - 1, :]  # modification de la ligne
            L[i][k - 1] = c  # on ajoute le coef du pivot dans la matrice L
    return L, U


# print(LU(A)[0])
# print(LU(A)[1])

# Questions 3
A = np.array([[2, 3, 5], [2, 2, 4], [2, 1, 4]])  # matrice test
L, U = LU(A)  # appel de la fonction de Decomposition LU
print(" Decomposition LU de la matrice A :")
print("A =")
for ligne in A:  # plus de code pour print mais plus jolie dans la console...
    print(ligne)
print("L =")
for ligne in L:
    print(ligne)
print("U =")
for ligne in U:
    print(ligne)

# Question 4
# verification
M = np.dot(L, U)  # dot product L*U, produit matricielle de L et U
print("Verification : A = LU ? \n L * U =")
for ligne in M:
    print(ligne)


# Exercice 2 : Resolution d'un systeme lineaire
# Question 1
def desrem(A, B):
    """Descente-remonte. Resoud AX=B.

    param A = array, matrice carree
    param B = array, matrice colonne
    return = array, X : solution du systeme lineaire.
    """
    L, U = LU(A)  # decomposition LU de A
    n, p = np.shape(A)  # dim de A
    Y = np.zeros(n)  # Initialisation de Y en vecteur colonne nul
    X = np.zeros(n)  # Initialisation de X en vecteur colonne nul
    q, r = np.shape(B)  # dim de B
    if r != 1:  # verif erreur dim de B
        print("Erreur : la matrice B doit etre une matrice colonne !")
        return None
    for i in range(n):  # descente
        s = 0  # Initialisation de notre somme
        for j in range(n):  # 2eme boucle for pour la somme
            s += L[i][j] * Y[j]  # somme des L_i,j * Y_j
        Y[i] = B[i] - s  # on soustrait s de b_i pour avoir y_i
    # remonter, on part d'en bas cad de n (n-1 python), jusqu'a 0 (-1 python), avec un pas de -1 (n-1, n-2, etc.)
    for i in range(n - 1, -1, -1):
        s = 0  # init somme
        for j in range(n - 1, i, -1):  # tjs en partant d'en bas, de n (n-1 python), a i+1 (i python)
            s += U[i][j] * X[j] / U[i][i]  # somme
        X[i] = Y[i] / U[i][i] - s  # on soustrait la somme comme dans la descente
    return X  # solution du systeme


# Question 2
# (a)
A = [[2, 4, 2],
     [1, 1, 1],
     [1, 1, -2]]

B = [[80],
     [30],
     [0]]


x = desrem(A, B)[0]
y = desrem(A, B)[1]
z = desrem(A, B)[2]
print("Le restaurateur va pouvoir acheter : \n {} kg d'aubergines \n {} kg de tomates \n {} kg de courgettes".format(x, y, z))
# print(np.linalg.solve(A, B))  # verification


# (b)
A = [[2, 3, 1],
     [1, 1, 1],
     [1, -2, 1]]

B = [[67],
     [30],
     [0]]

P = [[0, 0, 1],  # matrice de passage
     [0, 1, 0],
     [1, 0, 0]]


def desrem_P(A, B, P=np.identity(len(A))):  # modification pour eviter de diviser par zero
    """Descente-remonte. Resoud AX=B.

    param A = array, matrice carree
    param B = array, matrice colonne
    param P = array, matrice de passage
    return = array, X : solution du systeme lineaire.
    """
    PA = np.dot(P, A)
    A = np.dot(PA, P)
    B = np.dot(P, B)
    L, U = LU(A)  # decomposition LU de A
    n, p = np.shape(A)  # dim de A
    Y = np.zeros(n)  # Initialisation de Y en vecteur colonne nul
    X = np.zeros(n)  # Initialisation de X en vecteur colonne nul
    q, r = np.shape(B)  # dim de B
    if r != 1:  # verif erreur dim de B
        print("Erreur : la matrice B doit etre une matrice colonne !")
        return None
    for i in range(n):  # descente
        s = 0  # Initialisation de notre somme
        for j in range(n):  # 2eme boucle for pour la somme
            s += L[i][j] * Y[j]  # somme des L_i,j * Y_j
        Y[i] = B[i] - s  # on soustrait s de b_i pour avoir y_i
    # remonter, on part d'en bas cad de n (n-1 python), jusqu'a 0 (-1  python), avec un pas de -1 ( n-1, n-2 etc.)
    for i in range(n - 1, -1, -1):
        s = 0  # init somme
        for j in range(n - 1, i, -1):  # tjs en partant d'en bas, de n (n-1 python), a i+1 (i python)
            s += U[i][j] * X[j] / U[i][i]  # somme
        X[i] = Y[i] / U[i][i] - s  # on soustrait la somme comme dans la descente
    return np.dot(P, X)  # solution du systeme multiplier par la matrice de passage


x = desrem_P(A, B, P)[0]
y = desrem_P(A, B, P)[1]
z = desrem_P(A, B, P)[2]
print("Le restaurateur va pouvoir acheter : \n {} kg d'aubergines \n {} kg de tomates \n {} kg de courgettes".format(x, y, z))
# print(np.linalg.solve(A, B))  # verification
