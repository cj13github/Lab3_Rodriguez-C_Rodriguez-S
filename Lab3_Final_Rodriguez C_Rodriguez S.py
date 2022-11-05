import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from PIL import Image

ST="-"
if ST== "-":
    st.title("¡Bienvenido!")
    st.text('''
    Esta es una interfaz gráfica realizada por una serie de códigos diseñados por 
    estudiantes de la universidad del norte en la que podrás interactuar con diferentes 
    funciones aplicando la serie y transformada de Fourier en cualquiera de los casos 
    con el fin de analizar, observar, y conocer un poco más su estructura y comportamiento.
    ''')
    st.subheader("Por favor, seleccione si desea hacer series de Fourier o Transformada de Fourier")
    ST=st.selectbox("", options=["-","Series de Fourier", "Transformada de Fourier"] )



if ST== "Series de Fourier":

    #TITULO

    st.title("Series de Fourier")

    img = Image.open('serie1.png')
    st.image(img, width=700)

    st.latex(r'''
    X(t)=A₀+\sum_{k=1}^{\infty} [aₖ*Cos(\frac{2π}{T}kt)+bₖ*Sen(\frac{2π}{T}kt)])''')


    #MUESTRA LA FUNCIÓN QUE SELECCIONÓ EL USUARIO
    st.subheader("Por favor, selecciones el tipo de señal que desee recrear")
    tiposeñal=st.selectbox("", options=["Exponencial", "Triangular", "Cuadrada", "Senoidal rectificada", "Rampa trapezoidal", "cuadratica"] )
    if tiposeñal == "Exponencial":
        st.latex(r'''X(t)=Ae^{-bt}''')
    if tiposeñal == "Triangular":
        st.latex(r'''X(t)=A*sawtooth(\frac{2π}{T}t,0.5)''')
    if tiposeñal == "Cuadrada":
        st.latex(r'''X(t)=A*square(\frac{2π}{T}t,0.5)''')
    if tiposeñal == "Senoidal rectificada":
        st.latex(r'''X(t)=A|{Sen(Wt)}|''')
    if tiposeñal == "cuadratica":
        st.latex(r'''X(t)=t^2''')
    if tiposeñal == "Rampa trapezoidal":
        st.latex(r'''X₁(t)=t,[0<t<\frac {T} {3}]''') 
        st.latex(r'''X₂(t)=\frac {T} {3},[\frac {T} {3}<t<\frac {2T} {3}]''')  
        st.latex(r'''X₃(t)=-t, [\frac {2T} {3}<t<T]''')


    #RECOLECCIÓN DE LAS VARIABLES Y DECLARACIÓN DE VECTORES
    st.subheader("Por favor, introduzca el periodo [T]")
    T=st.number_input("",1, key="1") 
    dt=0.005
    m= int((0+2*T+dt)/dt) #Tamaño de la señal
    if tiposeñal == "Exponencial":
        t= np.arange(0,T+dt,dt) #Tamaño del vector
    if tiposeñal == "Triangular":
        t= np.arange(0,T+dt,dt) #Tamaño del vector
    if tiposeñal == "Cuadrada":
        t= np.arange(0,2*T+dt,dt) #Tamaño del vector
    if tiposeñal == "Senoidal rectificada":
        t= np.arange(0,2*T+dt,dt) #Tamaño del vector
    if tiposeñal == "cuadratica":
        t= np.arange(0,T+dt,dt) #Tamaño del vector
    if tiposeñal == "Rampa trapezoidal":
        t= np.arange(0,T,dt) #Tamaño del vector   
    st.subheader("Por favor, digite el número de armonicos [n]")
    n = st.number_input("",2, key ="2")  #Numero de ármonicos
    n=int(n)
    ak = [0]*n
    bk = [0]*n
    ak = np.zeros(n)
    bk = np.zeros(n)
    m=len(t) #Dimensión de la señal de salida
    a0=0
    pc=0


    #FUNCIÓN ORIGINAL X(t)
    if tiposeñal == "Triangular":
        st.subheader("Por favor, introduzca la amplitud [A]")
        A=st.number_input("",1) #Lectura de la amplitud
        w0=(2*np.pi)/T #Periodo en terminos de la frecuencia angular
        x=A*signal.sawtooth(w0*t,0.5) #Señal
        max1=A  
    if tiposeñal == "Cuadrada":
        st.subheader("Por favor, introduzca la amplitud [A]")
        A1=st.number_input("",1) #Lectura de la amplitud
        w0=(2*np.pi)/T #Periodo en terminos de la frecuencia angular 
        x=A1*signal.square(w0*t,0.5) #Señal
        max1=A1 #Calculo del máximo valor de la función
        pc=A1*0.35
    if tiposeñal == "Senoidal rectificada":
        st.subheader("Por favor, introduzca la amplitud [A]")
        A2=st.number_input("",1) #Lectura de la amplitud
        w0=(2*np.pi)/T #Periodo en terminos de la frecuencia angular
        x=A2*abs(np.sin(w0*t)) #Señal
        max1=A2 #Calculo del máximo valor de la función   
    if tiposeñal == "Exponencial":
        st.subheader("Por favor, digite el intercepto [a]")
        a=st.number_input("",1) #Lectura del intercepto
        st.subheader("Por favor, digite el factor de decrecimiento [b]")
        b=st.number_input("",1, key= "3") #Lectura del factor de decrecimiento    
        w0=(2*np.pi)/T #Periodo en terminos de la frecuencia angular
        x= a*(np.exp(-b*t))
        max1=a #Calculo del máximo valor de la función   
    if tiposeñal == "cuadratica":
        x=t*t
        w0=(2*np.pi)/T #Periodo en terminos de la frecuencia angular
        max1=T**2
    if tiposeñal == "Rampa trapezoidal":
       

        t=np.arange(0,T,dt)
        w0=(2*np.pi)/T
        xinicio = 0
        xfinal = T
        e=(xfinal-xinicio)/3
        def tramo1(z):         
            return z-xinicio    
        def tramo2(z):         
            return e    
        def tramo3(z):         
            return -z+xfinal     
        a=xinicio
        b=xinicio+e   
        c=xinicio+2*e
        d=xinicio+3*e

        x=np.piecewise(t,[(a<=t) & (t<b),(b<=t)&(t<=c),(c<t)&(t<=d)],[lambda t:tramo1(t),lambda t: tramo2(t),lambda t:tramo3(t)])    
        tramo1=np.vectorize(tramo1)     
        #graph.plot(t[t<b],tramo1(t[t<b]),c="c")  
        tramo2=np.vectorize(tramo2) 
        #graph.plot(t[(b<=t)&(t<c)],tramo2(t[(b<=t)&(t<c)]),c="c") 
        tramo3=np.vectorize(tramo3)     
        #graph.plot(t[(c<=t)&(t<=d)],tramo3(t[(c<=t)&(t<=d)]),c="c") 
        max1 = e
 

    
    #BOTÓN Y ECUACIÓN DE LA SERIE DE FOURIER
    clicked2 = st.button("Realizar serie de Fourier")


    #CALCULO DE a₀, aₖ y bₖ
    maxtotal1=0
    an = np.zeros(n)
    An = np.zeros(n)
    for i in range(1,m): #Calculo de a₀
        a0= a0+(1/T)*x[i]*dt
    for i in range (1,n,1): #Calculo de aₖ, bₖ, Aₙ y aₙ
        for g in range (1,m,1):
            ak[i]=ak[i]+((2/T)*x[g]*np.cos(i*t[g]*w0))*dt #Calculo de aₖ
            bk[i]=bk[i]+((2/T)*x[g]*np.sin(i*t[g]*w0))*dt #Calculo de bₖ
    
        An[i]=(((ak[i])**2)+((bk[i])**2))**(1/2) #Calculo del espectro de Amplitud Aₙ
        an[i]=np.arctan((bk[i])/(ak[i]))*(-1) #Calculo del espectro de Fase aₙ
        maxtotal=An[i]
        maxtotal1=maxtotal1+maxtotal

    #CALCULO Y GRAFFICACIÓN DE LA SERIE DE FOURIER XF(t), X(t), Aₙ y aₙ
    xf=a0
    t1=np.arange(0,2*T+dt,dt)
    tAf=np.arange(1,n+1,1)
    if clicked2:
        xf= xf+ak[1]*np.cos(1*w0*t1)+bk[1]*np.sin(1*w0*t1)

        fig=plt.figure(figsize=(8,8))
        ax2=fig.add_subplot(2,1,1)
        ax1=fig.add_subplot(2,1,2)
        ax1.set_title('Serie de Fourier')
        ax2.set_title('Señal original')
        ax1.set_xlabel("Tiempo [Seg]")
        ax1.set_ylabel("Amplitud [t]")
        ax2.set_xlabel("Tiempo [Seg]")
        ax2.set_ylabel("Amplitud [t]")
        ax1.plot(t1,xf) 
        ax2.plot(t,x) 
        ax1.grid(color='gray', linestyle='dotted', linewidth=1.5)      
        ax2.grid(color='gray', linestyle='dotted', linewidth=1.5)
        fig.tight_layout()
        plots1=st.pyplot(fig)
        fig3=plt.figure(figsize=(8,8))
        ax3=fig3.add_subplot(2,1,1)
        ax3.set_title('Espectro de Amplitud')
        ax3.set_xlabel("Armónicos")
        ax3.set_ylabel("Amplitud")
        ax3.stem(tAf,An)
        ax3.grid(color='gray', linestyle='dotted', linewidth=1.5)
        ax4=fig3.add_subplot(2,1,2)
        ax4.set_title('Espectro de Fase')
        ax4.set_xlabel("Armónicos")
        ax4.set_ylabel("...")
        ax4.stem(tAf,an)       
        ax4.grid(color='gray', linestyle='dotted', linewidth=1.5)
        fig3.tight_layout()
        st.pyplot(fig3)
        for i in range(2,n,1):
            xf= xf+ak[i]*np.cos(i*w0*t1)+bk[i]*np.sin(i*w0*t1)
            max2=max(xf)
            max2=max2-pc
            max3=max2/max1
            m=xf/max3

            fig=plt.figure(figsize=(8,8))

            ax2=fig.add_subplot(2,1,1)
            ax1=fig.add_subplot(2,1,2)
            ax1.set_title('Serie de Fourier')
            ax2.set_title('Señal original')
            ax1.set_xlabel("Tiempo [Seg]")
            ax1.set_ylabel("Amplitud [t]")
            ax2.set_xlabel("Tiempo [Seg]")
            ax2.set_ylabel("Amplitud [t]")
            ax1.plot(t1,m)  
            ax2.plot(t,x) 
            ax1.grid(color='gray', linestyle='dotted', linewidth=1.5)      
            ax2.grid(color='gray', linestyle='dotted', linewidth=1.5)                                  
            fig.tight_layout()
            plots1.pyplot(fig)

            
if ST== "Transformada de Fourier":
    
    st.title("Transformada de Fourier")
    st.latex(r'''
    X(w)=\int_{-\infty}^{\infty} (A₀*Sen(\frac {w₀n} {Fs}) + A₁*Cos(\frac {w₁n} {Fs}) + A₂*Sen(\frac {w₂n} {Fs}))*e^{-i2W₀t}dt''')

    img = Image.open('transformada2.png')
    st.image(img, width=700)

    st.subheader("Por favor, digite el número de muestras [Nm] ")
    N=int(st.number_input("",1, key='1'))
    st.subheader("Por favor digite la frecuencia de muestreo [Fs]")
    Fs=int(st.number_input("",1, key="2"))
    st.subheader("Por favor, digite la amplitud [A₀] ")
    A0=int(st.number_input("",1, key='3'))
    st.subheader("Por favor, digite la amplitud [A₁] ")
    A1=int(st.number_input("",1, key='4'))
    st.subheader("Por favor, digite la amplitud [A₂] ")
    A2=int(st.number_input("",1, key='5'))
    st.subheader("Por favor, digite la frecuencia [W₀] ")
    f0=int(st.number_input("",1, key='6'))
    st.subheader("Por favor, digite la frecuencia [W₁] ")
    f1=int(st.number_input("",1, key='7'))
    st.subheader("Por favor, digite la frecuencia [W₂] ")
    f2=int(st.number_input("",1, key='8'))
    w0=2*np.pi*f0
    w1=2*np.pi*f1
    w2=2*np.pi*f2
    n=np.arange(0,N)
    dt=1/Fs
    a=0
    b=1
    t=np.arange(a,b,dt)
    y=t**2
    clicked1 = st.button('Realizar transformada de Fourier')
    if clicked1:

        fig1=plt.figure(figsize=(7,7))

        fig2=plt.figure(figsize=(7,7))

        x = A0*np.sin(w0*n/Fs) + A1*np.cos(w1*n/Fs) + A2*np.sin(w2*n/Fs)
        x1=np.sin(2*np.pi*2500*n/Fs)
        x2=2*np.cos(2*np.pi*300*t)+np.sin(2*np.pi*50*t)
        f=np.fft.fft(x)
        freq=np.fft.fftfreq(len(x))*Fs

        ax1=fig1.add_subplot(2,1,1)
        ax2=fig1.add_subplot(2,1,2)
        ax2.set_title('Transformada de Fourier de la señal continua')
        ax1.set_title('Señal original continua')
        ax1.set_xlabel("Tiempo [Seg]")
        ax1.set_ylabel("Amplitud en el tiempo")
        ax2.set_xlabel("Frecuencia [Hz]")
        ax2.set_ylabel("Amplitud en la frecuencia")
        ax1.plot(n,x)
        ax2.plot(freq,abs(f))
        ax1.grid(color='gray', linestyle='dotted', linewidth=1.5)      
        ax2.grid(color='gray', linestyle='dotted', linewidth=1.5)                                  
        fig1.tight_layout()
        st.pyplot(fig1)
        
        ax3=fig2.add_subplot(2,1,1)
        ax3.set_title('Señal original discreta')
        ax3.set_xlabel("Tiempo [Seg]")
        ax3.set_ylabel("Amplitud en el tiempo")
        ax3.stem(n,x)
        ax3.grid(color='gray', linestyle='dotted', linewidth=1.5)
        ax4=fig2.add_subplot(2,1,2)
        ax4.set_title('Transformada de Fourier de la señal discreta')
        ax4.set_xlabel("Frecuencia [Hz]")
        ax4.set_ylabel("Amplitud en la frecuencia")
        ax4.stem(freq,abs(f))      
        ax4.grid(color='gray', linestyle='dotted', linewidth=1.5)
        fig2.tight_layout()
        st.pyplot(fig2)
        

        