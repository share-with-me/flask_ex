#Making use of matplotlib to plot based on input data
import numpy 
from numpy import *
import matplotlib.pyplot as plt
import os, time, glob
from math import pi

def compute(A,B):
    domain = eval(B, numpy.__dict__)
    #x = numpy.linspace(domain[0], domain[1], 100)
    #x = numpy.arange(domain[0],domain[1],0.01)
    x = numpy.array(domain)
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

if __name__ == '__main__':
    print compute('x**2','range(0,100)') #Makes sure that if compute.py is called directly, it will make a directory named static and the resultant png 
    #generated will be stored in that directory with default compute arguments - 1,0.1,1,20
    #pass