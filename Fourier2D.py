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


#PROCESO PARA QUITAR EL RUIDO DE LA IMAGEN
factor_c= 0.087
factor_f= 0.087

Fourier=Fourier_arbol.copy()

f,c=copia.shape


Fourier[int(f*factor_f):int(f*(1-factor_f))]=0

Fourier[int(c*factor_c):int(c*(1-factor_c))]=0

plt.figure()
plt.imshow(abs(Fourier),norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada filtrada')
plt.savefig("CarvajalJuan_FT2D_filtrada.pdf")


nueva=fftpack.ifft2(Fourier).real

plt.figure()
plt.imshow(nueva,plt.cm.gray)
plt.savefig("CarvajalJuan_Imagen_filtrada.pdf")
