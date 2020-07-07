# Tarea4
Estudiante: Andrea Pizarro Urroz
Carné: B65454

#### Parte 1: 
Crear un esquema de modulación BPSK para los bits presentados. Esto implica asignar una forma de onda sinusoidal normalizada (amplitud unitaria) para cada bit y luego una concatenación de todas estas formas de onda.
~~~python
import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
from scipy import integrate
import matplotlib.pyplot as plt

datos = pd.read_csv('bits10k.csv',names=['A'],header=None)
bits=datos['A']


#Frecuencia 
f=5000 #Hz

#Cantidad de bits a trabajar 
N=10000#

#Periodo
T=1/f 

#Numero de Puntos de muestreo
p=50

#Puntos de muestreo de cada periodo
tp=np.linspace(0,T,P)

#forma de onda de la portadora
seno =np.sin(2*np.pi*f*tp)

#Visualización de la función portadora 
plt.plot(tp,seno)
plt.figure(1)
plt.xlabel('Tiempo(s)')
plt.title('Función Portadora')
plt.savefig('Portadora.png')

#frecuencia de muestreo
fs= p/T 

#linea temporal para toda la senal
t = np.linspace(0,N*T,N*p)




#Inicializador
señal= np.zeros(t.shape)


#Creacion de la senal modulada BPSK
for k,b in enumerate(bits):
    if b==1:
        señal[k*p:(k+1)*p]= seno
    else:
        señal[k*p:(k+1)*p]=-seno
           
#Visualizacion
plt.figure(2)
plt.title("Señal Modulada")
plt.savefig("SeñalModulada.png")
plt.plot(senal[0:5*p])
plt.show()
~~~
![SeñalModulada](SeñalModulada.png)
#### Parte 2
Calcular la potencia promedio de la señal modulada generada:
~~~python
#potencia instantanea 
Pins=señal**2

#potencia promedio
Pr=integrate.trapz(Pins,t)/(N*T)
print('Potencia promedio:', Pr)
~~~
La respuesta es la siguiente:
Potencia promedio: 0.4900009800019598 

#### Parte 3
 Graficar la densidad espectral de potencia de la señal con el método de Welch (SciPy), ANTES del canal ruidoso.

~~~python
fw, PSD = signal.welch(senal, fs, nperseg=1024)
plt.figure(4)
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.show()
~~~
![densidadantes](densidadantes.png)
#### Parte 4
Simular un canal ruidoso del tipo AWGN (ruido aditivo blanco gaussiano) con una relación señal a ruido (SNR) desde -5 hasta 0 dB
~~~python
# Relación señal-a-ruido deseada
SNR = range(-2,4)
BER=[]
for snr in SNR:
    # Potencia del ruido para SNR y potencia de la señal dadas
    Pn = Pr / (10**(snr / 10))

    # Desviación estándar del ruido
    sigma = np.sqrt(Pn)

    # Crear ruido (Pn = sigma^2)
    ruido = np.random.normal(0, sigma, senal.shape)

    # Simular "el canal": señal recibida
    Sr = senal + ruido

    # Visualización de los primeros bits recibidos
    pb = 5 #primeros bits 
    plt.figure(3)
    plt.plot(Sr[0:pb*p])
    plt.show()
     # Pseudo-energía de la onda original (esta es suma, no integral)
Es = np.sum(seno**2)
    # Inicialización del vector de bits recibidos
bitsSr =np.zeros(bits.shape)

    # Decodificación de la señal por detección de energía
for i, b in enumerate(bits):
    Ep = np.sum(Sr[i*P:(i+1)*P] * seno)
    if Ep > Es/2:
        bitsSr[i] = 1
    else:
        bitsSr[i] = 0  
            
err = np.sum(np.abs(bits - bitsSr))
BER.append(err/N)
print('Hay un total de {} errores en {} bits para una tasa de error de {} para el SNR de {}'.format(err, N, err/N, snr))

~~~
![señalruidosa](señalruidosa.png)
#### Parte 5
Graficar la densidad espectral de potencia de la señal con el método de Welch (SciPy), DESPUÉS del canal ruidoso.
~~~python
fw, PSD = signal.welch(Sr, fs, nperseg=1024)
plt.figure(5)
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.show()
~~~
![densidaddespués](densidaddespués.png)
#### Parte 6
~~~python
 # Pseudo-energía de la onda original (esta es suma, no integral)
    Es = np.sum(seno**2)
    # Inicialización del vector de bits recibidos
    bitsSr = np.zeros(bits.shape)

    # Decodificación de la señal por detección de energía
    for i, b in enumerate(bits):
        Ep = np.sum(Sr[i*P:(i+1)*P] * seno)
        if Ep > Es/2:
            bitsSr[i] = 1
        else:
            bitsSr[i] = 0  
            
    err = np.sum(np.abs(bits - bitsSr))
    BER.append(err/N)
    print('Hay un total de {} errores en {} bits para una tasa de error de {} para el SNR de {}'.format(err, N, err/N, snr))
~~~
![-2](-2.png)
![-1](-1.png)
![0](0.png)
![1](1.png)
![2](2.png)

#### Parte 7
~~~python
plt.semilogy(SNR, BER)
plt.title("VERSUS ")
plt.xlabel('SNR')
plt.show()
~~~
![BERSNR](BERSNR.png)   
    
