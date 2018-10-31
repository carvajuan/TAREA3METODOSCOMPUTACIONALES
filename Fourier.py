import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate

temp=open("signal.dat","r")
temp1=open("incompletos.dat","r")

file=open("signal.dat","r")
file1=open("incompletos.dat","r")

tamaño_signal=0
for linea in temp:
    linea=linea.split(",")
    tamaño_signal=tamaño_signal+1
    
sig_y=np.zeros(tamaño_signal)
sig_x=np.zeros(tamaño_signal)

tamaño_incompletos=0

for linea in temp1:
    linea=linea.split(",")
    tamaño_incompletos=tamaño_incompletos+1
    
incompletos_x=np.zeros(tamaño_incompletos)
incompletos_y=np.zeros(tamaño_incompletos)

contador=0

for lin in file:
    lin=lin.split(",")
    sig_y[contador]=lin[1]
    sig_x[contador]=lin[0]
    contador=contador+1
contador1=0
for l in file1:
    l=l.split(",")
    incompletos_y[contador1]=l[1]
    incompletos_x[contador1]=l[0]
    contador1=contador1+1

#TODOS LOS CICLOS ANTES DE ESTE COMENTARIO SON PARA ALMACENAR TODOS LOS DATOS NECESARIOS DE LOS ARCHIVOS
 
    
    
def Freq(n, d = 1.0): #FUNCION PARA CALCULAR LAS FRECUENCIAS DE LA TRANSFORMADA
    f = np.arange(0, n//2 + 1)/(d*n)
    return f


def Fourier(data): #IMPLEMENTACION PROPRIA PARA CALCULAR LA TRANSFORMADA DE FOURIER
    N = len(data)
    n = np.arange(0, N)
    trans = np.zeros(N//2 + 1, dtype=complex)
    for k in range(trans.shape[0]):
        trans[k] = (data*np.exp(-1j*2*np.pi*k*n/N)).sum()
    return trans

def pasabajos(info, frecuencias, corte): #FUNCION PARA HACER FILTRO PASABAJOS
    filtro = info.copy()
    filtro[frecuencias > corte] = 0
    return filtro

plt.plot(sig_x,sig_y,label="Signal") #GRAFICA DE LOS DATOS SIGNAL
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title("Grafica datos siganl.dat")
plt.savefig("CarvajalJuan_signal.pdf")


Transformada_propia=Fourier(sig_y)   #CALCULO DE TRANSFORMADA
Frecuencias_propias=Freq(512, d=1.0/17955)   
Transformada_paquete= np.fft.rfft(sig_y) 
Frecuencias_paquete = np.fft.rfftfreq(512, d=1./17955)

print("NO USÉ EL PAQUETE DE FRECUENCIAS DE NUMPY")

plt.plot(Frecuencias_propias,abs(Transformada_propia),label="Transformada Signal")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim([0,2500])
plt.legend()
plt.title("Transformada Fourier signal.dat")
plt.savefig("CarvajalJuan_TF.pdf")


def principales(): #FUNCION PARA SACAR LAS FRECUENCIAS PRINCIPALES DE LA TRANSFORMADA
    conta=0
    Principales=np.zeros(4)
    for i in range(len(Frecuencias_propias)):
        if(Transformada_propia[i]>100):
            Principales[conta]=Frecuencias_propias[i]
            conta=conta+1
    return Principales  

print("Las frecuencias principales de los datos signal son: ",principales())


Primer_filtro=pasabajos(Transformada_propia,Frecuencias_propias,1000)

Inversa_filtro=np.fft.irfft(Primer_filtro)

plt.plot(sig_x,Inversa_filtro,label="Inversa con filtro")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title("Inversa con filtro de 1000 Hz")
plt.savefig("CarvajalJuan_filtrada.pdf")


print("Si se puede 'hacer' la transformada de fourier de los datos incompletos pero estos nos da valores negativos lo cual carece de significado fisico y no tiene sentido en nuestro mundo")


Interpolacion_cubica= interpolate.interp1d(incompletos_x, incompletos_y,kind="cubic")

Interpolacion_cuadratica= interpolate.interp1d(incompletos_x, incompletos_y,kind="quadratic")


x_new=np.linspace(0.000390625,0.028515625,512)
Cubic=Interpolacion_cubica(x_new)
Quad=Interpolacion_cuadratica(x_new)

Fourier_Cubic=Fourier(Cubic)
Fourier_Quad=Fourier(Quad)


plt.figure()
plt.subplot(311)
plt.plot(Frecuencias_propias,abs(Transformada_propia),c="red",label="T.Signal")
plt.xlabel("Frecuencia")
plt.xlim([0,5000])
plt.legend()
plt.subplot(312)
plt.plot(Frecuencias_propias,abs(Fourier_Cubic),c="green",label="T.Cubica")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada de Fourier")
plt.xlim([0,5000])
plt.legend()
plt.subplot(313)
plt.plot(Frecuencias_propias,abs(Fourier_Quad),c="blue",label="T.Cuadratica")
plt.xlabel("Frecuencia")
plt.xlim([0,5000])
plt.legend()
plt.savefig("CarvajalJuan_TF_interpola.pdf")
print("Las diferencias entre las transformadas de fourier de los distintos datos se puede observar como un ruido, las interpolaciones tienen un ruido mas notable que los datos signal")
print("Mientras mayor sea el grado de la regresion que se haga en los datos incompletos su transformada se acercara mas a la de los datos signal, ya que los datos incompletos.dat como es obvio son una version incompleta de los datos signal.dat")



Filtro_500_signal=np.fft.irfft(pasabajos(Transformada_propia,Frecuencias_propias,500))
Filtro_500_cubic=np.fft.irfft(pasabajos(Fourier_Cubic,Frecuencias_propias,500))
Filtro_500_quad=np.fft.irfft(pasabajos(Fourier_Quad,Frecuencias_propias,500))
Filtro_1000_cubic=np.fft.irfft(pasabajos(Fourier_Cubic,Frecuencias_propias,1000))
Filtro_1000_quad=np.fft.irfft(pasabajos(Fourier_Quad,Frecuencias_propias,1000))

plt.subplot(211)
plt.plot(sig_x,Inversa_filtro,c="blue",label="signal-filtro 1000-Hz")
plt.plot(x_new,Filtro_1000_quad,c="red",label="Cuadratica-filtro 1000-Hz")
plt.plot(x_new,Filtro_1000_cubic,c="green",label="Cubica-filtro 1000-Hz")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.subplot(212)
plt.plot(sig_x,Filtro_500_signal,c="blue",label="signal-filtro 500-Hz")
plt.plot(x_new,Filtro_500_quad,c="red",label="Cuadratica-filtro 500-Hz")
plt.plot(x_new,Filtro_500_cubic,c="green",label="Cubica-filtro 500-Hz")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("CarvajalJuan_2Filtros.pdf")
