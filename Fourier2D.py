import numpy as np
import matplotlib.pylab as plt
from scipy import fftpack
from scipy import ndimage
from PIL import Image
from matplotlib.colors import LogNorm

arbol = plt.imread('arbol.png').astype(float) #SE LEE Y ALMACENA LA IMAGEN COMO UN ARRAY DE FLOATS

Fourier_arbol=fftpack.fft2(arbol) #Transformada de fourier de la imagen

plt.figure()
plt.imshow(Fourier_arbol)
plt.title("Imagen original")
plt.savefig("Original.png")

plt.figure()
plt.imshow(abs(Fourier_arbol),norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de la imagen Arbol')
plt.savefig("CarvajalJuan_FT2D.pdf")
