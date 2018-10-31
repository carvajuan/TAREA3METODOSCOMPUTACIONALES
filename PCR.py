import matplotlib.pyplot as plt
import numpy as np
import urllib.request

#SE DESCARGA EL ARCHIVO DE INTERNET
urllib.request.urlretrieve("http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat", "WDBC.dat")

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
            l[i]=1.0
            dat[contador,i]=l[i]
        elif(l[i]=="B"):  #ALMACENAR ESTOS DATOS EN BINARIO ME ESTABA DANDO PROBLEMAS ASI QUE DECIDI DARLES ESTOS VALORES
            l[i]=2.0
            dat[contador,i]=l[i]
        else:
            dat[contador,i]=l[i]
    contador=contador+1
    
    
datas = np.delete(dat, 0, 1) #ELIMINA LA COLUMNA DE LOS IDS
D=datas.copy()
for i in range(31): #SE NORMALIZAN LAS COLUMNAS DE LOS DATOS
    datas[:,i]=(datas[:,i]-np.mean(datas[:,i]))/(np.sqrt(np.var(datas[:,i])))
print(datas) 

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

auto_valores,auto_vectores=np.linalg.eig(covarianza(datas))

print("La matriz de covarianza sacada por mi cuenta da, despues de normalizar las columnas de los datos y quitar la columna de IDS:")
print(covarianza(datas))

for i in range(len(auto_valores)): #SE IMPRIMEN LOS AUTOVALORES Y VECTORES DE LA MATRIZ DE COVARIANZA
    print("El autovalor",auto_valores[i]," tiene asociado el autovector",auto_vectores[:,i])

vector_1=auto_vectores[:,0]
vector_2=auto_vectores[:,1]  
print("Los dos autovalores mas importantes son:",auto_valores[0], auto_valores[1],"y los autovectores asociados son", vector_1,vector_2)

PC1=np.dot(datas, vector_1)/np.linalg.norm(vector_1)
PC2=np.dot(datas, vector_2)/np.linalg.norm(vector_2)

M=[]
B=[]
X=[]
Y=[]

for i in range(len(D[:,0])):
    
    if(D[i,0]==1.0):
        
        M.append(PC1[i])
        X.append(PC2[i])
        
    else:
        B.append(PC1[i])
        Y.append(PC2[i])

    
plt.figure()
plt.scatter(X,M, c="red",label="Malignos")
plt.scatter(Y,B, c="green",label="Benignos")
plt.legend()
plt.savefig("CarvajalJuan_PCA.pdf")

print("En este caso el metodo de PCA si es util para analizar y predecir si un tumor es benigno o maligno, sin embargo no es 100% confiable, como se puede ver en la grafica, algunos puntos se sobrelapan lo que significa que en algunas ocaciones ete metodo fallara al identificar la naturaleza del tumor. Pero puedo decir que para un diagnostico temprano este metodo puede ser util")
