from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy.special as sp

tacheDiffraction = Image.open("disque1mm.png")

tacheDiffractionGris = tacheDiffraction.convert("L")
data = np.asarray(tacheDiffractionGris)
hauteur = data.shape[0]
longueur = data.shape[1]

taillePixel = 5.3*10**(-6)
D = 80*10**(-2)
a = 1*10**(-3) #diamètre de la pupille ET PAS LE RAYON, VERIFIER DANS CR
LAMBDA = 632.8*10**(-9)
ALPHA =((m.pi * a)/(LAMBDA*D)) #cf commentaire d'après pour l'explication sur ce alpha

tailleCache = 25

maxHauteur = np.where(data == np.max(data))[0][0]
maxLongueur = np.where(data == np.max(data))[1][0]

#Crééer un cache

cacheImage = data[maxHauteur - tailleCache // 2 : maxHauteur + tailleCache // 2 + 1, :]


valeurPixel = np.mean(cacheImage, axis=0)/np.max(np.mean(cacheImage, axis=0)) #courbe moyennée
listeNumero = np.linspace(0, longueur, longueur) #[0, 1, 2, ..., n]
listeY = list(map(lambda x: x*taillePixel, listeNumero)) #on veut l'abcisse y sur l'axe, je tranforme en micromètre
listeYCentree = list(map(lambda x: x - maxLongueur*taillePixel, listeY)) #bessel a un max en 0, je centre le max de ma tache en 0
listeYCentreeALPHA = list(map(lambda x: x*ALPHA, listeYCentree)) #pour avoir Pi*x comme dans l'énoncé de tp, mais x = a*theta/lambda et theta = y/D
# d'où Pi*x = Pi*a*y/D*lambda = ALPHA*y

EPSILON = [] #Epsilon = J(alpha y)/(alpha y)**2 avec alpha = (pi*a)/(lambda*D) et y position sur l'écran, max centré en 0
J1 = [] #liste des valeurs de la fonction de Bessel de premier ordre

J1  = sp.j1(1,np.array(listeYCentree)) #donne la liste des valeurs des J1

for i in range(len(listeYCentree)): #boucle for provisoire :)
    EPSILON.append((J1[i]/listeYCentreeALPHA[i])**2) #cf formule du tp: epsilon = (J(pi*x)/pi*x) **2

plt.plot(listeYCentree, valeurPixel, label="Courbe centrée")
plt.plot(listeYCentree, EPSILON, label="Bessel")
plt.ylim(0, 1)
plt.legend()
plt.title("Valeurs du niveau de gris des pixels normalisée à l'unitée")
plt.xlabel("Position (micromètre)")
plt.ylabel("Niveau de gris")
plt.show()
