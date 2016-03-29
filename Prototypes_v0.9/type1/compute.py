#Making use of matplotlib to plot based on input data
from numpy import exp, cos, linspace
import matplotlib.pyplot as plt
import os, time, glob

def damped_vibrations(t, A, b, w):
    return A*exp(-b*t)*cos(w*t) #Function for damped vibrations

def compute(A, b, w, T, resolution=500):
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

if __name__ == '__main__':
    print compute(1, 0.1, 1, 20) #Makes sure that if compute.py is called directly, it will make a directory named static and the resultant png 
    #generated will be stored in that directory with default compute arguments - 1,0.1,1,20