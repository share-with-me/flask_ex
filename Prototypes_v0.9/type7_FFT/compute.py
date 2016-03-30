#Making use of matplotlib to plot based on input data
import numpy as np
from numpy import exp, cos, linspace
import matplotlib.pyplot as plt
import os, time, glob
import plotly.plotly as py
from scipy.fftpack import dct, idct
from scipy.fftpack import dst, idst
from scipy.fftpack import ifftn,ifft2
import matplotlib.cm as cm

def func(ff,t):
    return np.sin(50.0*2.0*np.pi*ff*t) + 0.5*np.sin(80.0*2.0*np.pi*ff*t) 

def damped_vibrations(t, A, b, w):
    return A*exp(-b*t)*cos(w*t) #Function for damped vibrations

def compute_Damping(A, b, w, T, resolution=500):
    t = linspace(0, T, resolution+1)
    u = damped_vibrations(t, A, b, w)
    plt.figure()  # needed to avoid adding curves in plot
    plt.plot(t, u)
    plt.title('A=%g, b=%g, w=%g' % (A, b, w)) #Title of the plot
    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    plotfile = os.path.join('static', str(time.time()) + '.png') #Name of file - unique and uncached by the browser
    plt.savefig(plotfile)
    return plotfile


def compute_1D(Fs, ff):
    fs = Fs
    ts = 1.0/fs
    t = np.arange(0, 1, ts)
    freq = ff
    y = func(freq, t)

    n = len(y) #length of the signal
    k  =np.arange(n)
    T = n/fs
    frq = k/T #two sides frequency range

    frq = frq[range(n/2)] #one side frequency range

    Y = np.fft.fft(y)/n #FFT computation and normalisation
    Y = Y[range(n/2)]
    plt.figure()  # needed to avoid adding curves in plot
    fig, x = plt.subplots(2, 1)
    x[0].plot(t,y)
    x[0].set_xlabel('Time')
    x[0].set_ylabel('Amplitude')
    x[1].plot(frq,abs(Y),'r') # plotting the spectrum
    x[1].set_xlabel('Freq (Hz)')
    x[1].set_ylabel('|Y(freq)|')
    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    plotfile = os.path.join('static', str(time.time()) + '.png') #Name of file - unique and uncached by the browser
    plt.savefig(plotfile)
    return plotfile

def compute_formula(A,B):
    domain = eval(B, np.__dict__)
    #x = numpy.linspace(domain[0], domain[1], 100)
    #x = numpy.arange(domain[0],domain[1],0.01)
    x = np.array(domain)
    t = eval(A)
    plt.figure()  # needed to avoid adding curves in plot
    plt.plot(x, t)
    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    plotfile = os.path.join('static', str(time.time()) + '.png') #Name of file - unique and uncached by the browser
    plt.savefig(plotfile)
    return plotfile

def compute_DCT(N):
    t = np.linspace(0,20,N)
    x = np.exp(-t/3)*np.cos(2*t)
    y = dct(x, norm='ortho')
    window = np.zeros(N)
    window[:20] = 1
    yr = idct(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    plt.plot(t, x, '-bx')
    plt.plot(t, yr, 'ro')
    window = np.zeros(N)
    window[:15] = 1
    yr = idct(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    # 0.0718818065008
    plt.plot(t, yr, 'g+')
    plt.legend(['x', '$x_{20}$', '$x_{15}$'])
    plt.show()


def compute_IDCT(N):
    t = np.linspace(0,20,N)
    x = np.exp(-t/3)*np.cos(2*t)
    y = dct(x, norm='ortho')
    window = np.zeros(N)
    window[:20] = 1
    yr = idct(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    plt.plot(t, x, '-bx')
    plt.plot(t, yr, 'ro')
    window = np.zeros(N)
    window[:15] = 1
    yr = idct(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    # 0.0718818065008
    plt.plot(t, yr, 'g+')
    plt.legend(['x', '$x_{20}$', '$x_{15}$'])
    plt.show()

def compute_DST(N):
    t = np.linspace(0,20,N)
    x = np.exp(-t/3)*np.cos(2*t)
    y = dst(x, norm='ortho')
    window = np.zeros(N)
    window[:20] = 1
    yr = idst(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    plt.plot(t, x, '-bx')
    plt.plot(t, yr, 'ro')
    window = np.zeros(N)
    window[:15] = 1
    yr = idst(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    # 0.0718818065008
    plt.plot(t, yr, 'g+')
    plt.legend(['x', '$x_{20}$', '$x_{15}$'])
    plt.show()

def compute_IDST(N):
    t = np.linspace(0,20,N)
    x = np.exp(-t/3)*np.cos(2*t)
    y = dst(x, norm='ortho')
    window = np.zeros(N)
    window[:20] = 1
    yr = idst(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    plt.plot(t, x, '-bx')
    plt.plot(t, yr, 'ro')
    window = np.zeros(N)
    window[:15] = 1
    yr = idst(y*window, norm='ortho')
    sum(abs(x-yr)**2) / sum(abs(x)**2)
    # 0.0718818065008
    plt.plot(t, yr, 'g+')
    plt.legend(['x', '$x_{20}$', '$x_{15}$'])
    plt.show()


def compute_2D(N):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    xf = np.zeros((N,N))
    xf[0, 5] = 1
    xf[0, N-5] = 1
    Z = ifftn(xf)
    ax1.imshow(xf, cmap=cm.Reds)
    ax4.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 0] = 1
    xf[N-5, 0] = 1
    Z = ifftn(xf)
    ax2.imshow(xf, cmap=cm.Reds)
    ax5.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 10] = 1
    xf[N-5, N-10] = 1
    Z = ifftn(xf)
    ax3.imshow(xf, cmap=cm.Reds)
    ax6.imshow(np.real(Z), cmap=cm.gray)
    plt.show()

def compute_ND(N):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    xf = np.zeros((N,N))
    xf[0, 5] = 1
    xf[0, N-5] = 1
    Z = ifftn(xf)
    ax1.imshow(xf, cmap=cm.Reds)
    ax4.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 0] = 1
    xf[N-5, 0] = 1
    Z = ifftn(xf)
    ax2.imshow(xf, cmap=cm.Reds)
    ax5.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 10] = 1
    xf[N-5, N-10] = 1
    Z = ifftn(xf)
    ax3.imshow(xf, cmap=cm.Reds)
    ax6.imshow(np.real(Z), cmap=cm.gray)
    plt.show()