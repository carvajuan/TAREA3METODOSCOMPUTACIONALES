import numpy as np
import matplotlib.pylab as plt
from scipy import fftpack
from scipy import ndimage
from PIL import Image
from matplotlib.colors import LogNorm

arbol = plt.imread('arbol.png').astype(float) #SE LEE Y ALMACENA LA IMAGEN COMO UN ARRAY DE FLOATS

F=np.fft.fft2(arbol) #Transformada de fourier de la imagen
Fourier_arbol=np.fft.fftshift(F)
Frecuencias_1=np.fft.fftfreq(len(Fourier_arbol[0])) #Arreglos de las frecuencias para saber cuales son las causantes del ruido
Frecuencias_2=np.fft.fftfreq(len(Fourier_arbol[1]))

plt.figure()
plt.imshow(arbol)
plt.title("Imagen original")
plt.savefig("Original.png")

plt.figure()
plt.imshow(np.log10(abs(Fourier_arbol)), cmap="gray")
plt.colorbar()
plt.title('Transformada de la imagen Arbol')
plt.savefig("CarvajalJuan_FT2D.pdf")

Fourier=Fourier_arbol.copy()
#Al graficar abs(Fourier_arbol) vs Frecuencias_1 y Frecuencias_2 observé que existen cuetro picos que representan el ruido periodico
#Ese ruido se ve en la imagen de la transformada como esos dos puntos blancos en los cuadrantes 2 y 4 
#Para quitar esos picos debo recorrer la matriz de la transformada de fourier de la imagen y establecer unas condiciones 
#Las condicones para quitar estos picos fueron resultado de prueba y error, sabia que debian tener menos de 5000 de magnitud pero no sabia muy bien donde cortarlos en la parte de abajo
for i in range(np.shape(Fourier_arbol)[0]):
  for j in range(np.shape(Fourier_arbol)[1]):
    if(abs(Fourier[i,j])>4100 and abs(Fourier[i,j])<5000):
      Fourier[i,j]=0.1  #Pongo 0.1 en vez de 0.0 para que cuando saque el logaritmo no me salga error (la imagen sale igual de bien)
      
plt.figure()
plt.imshow(np.log10(abs(Fourier)), cmap="gray")
plt.colorbar()
plt.title("Transformada filtrada")
plt.savefig("CarvajalJuan_FT2D_filtrada.pdf")

     
nueva=np.fft.ifft2(Fourier)

plt.figure()
plt.imshow(abs(nueva),cmap="gray")
plt.title("Imagen sin ruido")
plt.savefig("CarvajalJuan_Imagen_filtrada.pdf")
