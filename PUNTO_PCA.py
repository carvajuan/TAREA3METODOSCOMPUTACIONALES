import matplotlib.pyplot as plt
import numpy as np

datos=open("WDBC.dat")
atos=open("WDBC.dat")

tamaño=0
for linea in datos:  #CICLO PARA SABER LA CANTIDAD DE DATOS EN EL ARCHIVO
    tamaño=tamaño+1
    
dat=np.ones([tamaño,32])  #MATRIZ PARA LUEGO ALMACENAR LOS DATOS

contador=0

for l in atos:     #CICLO QUE RECORRE EL ARCHIVO DE DATOS Y ALMACENA SUS DATOS EN LA MATRIZ dat
    l=l.split(",")
    for i in range(32):
        if(l[i]=="M"):
            l[i]=1
        elif(l[i]=="B"):
            l[i]=0
        else:
            dat[contador,i]=l[i]
    contador=contador+1
  
#FUNCION PARA CALCULAR LA MATRIZ DE COVARIANZA 
def covarianza(datos):
    numero_columnas=np.shape(datos)[1]

    cantidad_datos=np.shape(datos)[0]

    matriz=np.ones([numero_columnas,numero_columnas])
    for i in range(numero_columnas):
        for j in range(numero_columnas):
            promedio_i=np.mean(datos[:,i])
            promedio_j=np.mean(datos[:,j])
            matriz[i,j]=(np.sum((datos[:,i]-promedio_i)*(datos[:,j]-promedio_j)))/(cantidad_datos-1.0)
    return matriz

auto_valores,auto_vectores=np.linalg.eig(covarianza(dat)))

print("La matriz de covarianza sacada por mi cuenta da")
print(covarianza(dat))
