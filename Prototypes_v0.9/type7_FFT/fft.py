#Making use of matplotlib to plot based on input data
import numpy as np
from numpy import exp, cos, linspace
import matplotlib.pyplot as plt
import os, time, glob
import plotly.plotly as py

def func(ff,t):
    return np.sin(50.0*2.0*np.pi*ff*t) + 0.5*np.sin(80.0*2.0*np.pi*ff*t) 

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

if __name__ == '__main__':
    print compute(150,5) #Makes sure that if compute.py is called directly, it will make a directory named static and the resultant png 
    #generated will be stored in that directory with default compute arguments - 150,5