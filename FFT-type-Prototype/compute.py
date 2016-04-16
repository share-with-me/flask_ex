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
from astroML.datasets import fetch_rrlyrae_templates


def compute_Convolution():

    from scipy.signal import fftconvolve

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Generate random x, y with a given covariance length
    np.random.seed(1)
    x = np.linspace(0, 1, 500)
    h = 0.01
    C = np.exp(-0.5 * (x - x[:, None]) ** 2 / h ** 2)
    y = 0.8 + 0.3 * np.random.multivariate_normal(np.zeros(len(x)), C)

#------------------------------------------------------------
# Define a normalized top-hat window function
    w = np.zeros_like(x)
    w[(x > 0.12) & (x < 0.28)] = 1

#------------------------------------------------------------
# Perform the convolution
    y_norm = np.convolve(np.ones_like(y), w, mode='full')
    valid_indices = (y_norm != 0)
    y_norm = y_norm[valid_indices]

    y_w = np.convolve(y, w, mode='full')[valid_indices] / y_norm

# trick: convolve with x-coordinate to find the center of the window at
#        each point.
    x_w = np.convolve(x, w, mode='full')[valid_indices] / y_norm

#------------------------------------------------------------
# Compute the Fourier transforms of the signal and window
    y_fft = np.fft.fft(y)
    w_fft = np.fft.fft(w)

    yw_fft = y_fft * w_fft
    yw_final = np.fft.ifft(yw_fft)

#------------------------------------------------------------
# Set up the plots
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.95,
                    hspace=0.05, wspace=0.05)

#----------------------------------------
# plot the data and window function
    ax = fig.add_subplot(221)
    ax.plot(x, y, '-k', label=r'data $D(x)$')
    ax.fill(x, w, color='gray', alpha=0.5,
        label=r'window $W(x)$')
    ax.fill(x, w[::-1], color='gray', alpha=0.5)

    ax.legend()
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    ax.set_ylabel('$D$')

    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0, 2.0)

#----------------------------------------
# plot the convolution
    ax = fig.add_subplot(223)
    ax.plot(x_w, y_w, '-k')

    ax.text(0.5, 0.95, "Convolution:\n" + r"$[D \ast W](x)$",
        ha='center', va='top', transform=ax.transAxes,
        bbox=dict(fc='w', ec='k', pad=8), zorder=2)

    ax.text(0.5, 0.05,
        (r'$[D \ast W](x)$' +
         r'$= \mathcal{F}^{-1}\{\mathcal{F}[D] \cdot \mathcal{F}[W]\}$'),
        ha='center', va='bottom', transform=ax.transAxes)

    for x_loc in (0.2, 0.8):
        y_loc = y_w[x_w <= x_loc][-1]
        ax.annotate('', (x_loc, y_loc), (x_loc, 2.0), zorder=1,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$D_W$')

    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0, 1.99)

#----------------------------------------
# plot the Fourier transforms
    N = len(x)
    k = - 0.5 * N + np.arange(N) * 1. / N / (x[1] - x[0])

    ax = fig.add_subplot(422)
    ax.plot(k, abs(np.fft.fftshift(y_fft)), '-k')

    ax.text(0.95, 0.95, r'$\mathcal{F}(D)$',
        ha='right', va='top', transform=ax.transAxes)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-5, 85)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax = fig.add_subplot(424)
    ax.plot(k, abs(np.fft.fftshift(w_fft)), '-k')

    ax.text(0.95, 0.95,  r'$\mathcal{F}(W)$', ha='right', va='top',
        transform=ax.transAxes)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-5, 85)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

#----------------------------------------
# plot the product of Fourier transforms
    ax = fig.add_subplot(224)
    ax.plot(k, abs(np.fft.fftshift(yw_fft)), '-k')

    ax.text(0.95, 0.95, ('Pointwise\nproduct:\n' +
                     r'$\mathcal{F}(D) \cdot \mathcal{F}(W)$'),
        ha='right', va='top', transform=ax.transAxes,
        bbox=dict(fc='w', ec='k', pad=8), zorder=2)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 3500)

    ax.set_xlabel('$k$')

    ax.yaxis.set_major_formatter(plt.NullFormatter())

#------------------------------------------------------------
# Plot flow arrows
    ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)

    arrowprops = dict(arrowstyle="simple",
                  color="gray", alpha=0.5,
                  shrinkA=5, shrinkB=5,
                  patchA=None,
                  patchB=None,
                  connectionstyle="arc3,rad=-0.35")

    ax.annotate('', [0.57, 0.57], [0.47, 0.57],
            arrowprops=arrowprops,
            transform=ax.transAxes)
    ax.annotate('', [0.57, 0.47], [0.57, 0.57],
            arrowprops=arrowprops,
            transform=ax.transAxes)
    ax.annotate('', [0.47, 0.47], [0.57, 0.47],
            arrowprops=arrowprops,
            transform=ax.transAxes)

    plt.show()
    
def compute_FFT():

    from scipy.fftpack import fft
    from scipy.stats import norm

    from astroML.fourier import PSD_continuous

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Draw the data
    np.random.seed(1)

    tj = np.linspace(-25, 25, 512)
    hj = np.sin(tj)
    hj *= norm(0, 10).pdf(tj)

#------------------------------------------------------------
# plot the results
    fig = plt.figure(figsize=(5, 3.75))
    fig.subplots_adjust(hspace=0.25)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    offsets = (0, 0.15)
    colors = ('black', 'gray')
    linewidths = (1, 2)
    errors = (0.005, 0.05)

    for (offset, color, error, linewidth) in zip(offsets, colors,
                                             errors, linewidths):
    # compute the PSD
        err = np.random.normal(0, error, size=hj.shape)
        hj_N = hj + err + offset
        fk, PSD = PSD_continuous(tj, hj_N)

    # plot the data and PSD
        ax1.scatter(tj, hj_N, s=4, c=color, lw=0)
        ax1.plot(tj, 0 * tj + offset, '-', c=color, lw=1)
        ax2.plot(fk, PSD, '-', c=color, lw=linewidth)

# vertical line marking the expected peak location
    ax2.plot([0.5 / np.pi, 0.5 / np.pi], [-0.1, 1], ':k', lw=1)

    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-0.1, 0.3001)

    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$h(t)$')

    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    ax2.set_xlim(0, 0.8)
    ax2.set_ylim(-0.101, 0.801)

    ax2.set_xlabel('$f$')
    ax2.set_ylabel('$PSD(f)$')

    plt.show()

def compute_Gaussian():


    from astroML.fourier import\
        FT_continuous, IFT_continuous, sinegauss, sinegauss_FT, wavelet_PSD

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Sample the function: localized noise
    np.random.seed(0)

    N = 1024
    t = np.linspace(-5, 5, N)
    x = np.ones(len(t))

    h = np.random.normal(0, 1, len(t))
    h *= np.exp(-0.5 * (t / 0.5) ** 2)

#------------------------------------------------------------
# Compute an example wavelet
    W = sinegauss(t, 0, 1.5, Q=1.0)

#------------------------------------------------------------
# Compute the wavelet PSD
    f0 = np.linspace(0.5, 7.5, 100)
    wPSD = wavelet_PSD(t, h, f0, Q=1.0)

#------------------------------------------------------------
# Plot the results
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0.05, left=0.12, right=0.95, bottom=0.08, top=0.95)

# First panel: the signal
    ax = fig.add_subplot(311)
    ax.plot(t, h, '-k', lw=1)
    ax.text(0.02, 0.95, ("Input Signal:\n"
                     "Localized Gaussian noise"),
        ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.9, 2.9)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylabel('$h(t)$')

# Second panel: an example wavelet
    ax = fig.add_subplot(312)
    ax.plot(t, W.real, '-k', label='real part', lw=1)
    ax.plot(t, W.imag, '--k', label='imag part', lw=1)

    ax.text(0.02, 0.95, ("Example Wavelet\n"
                     "$t_0 = 0$, $f_0=1.5$, $Q=1.0$"),
        ha='left', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.05,
        (r"$w(t; t_0, f_0, Q) = e^{-[f_0 (t - t_0) / Q]^2}"
         "e^{2 \pi i f_0 (t - t_0)}$"),
        ha='right', va='bottom', transform=ax.transAxes)

    ax.legend(loc=1)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_ylabel('$w(t; t_0, f_0, Q)$')
    ax.xaxis.set_major_formatter(plt.NullFormatter())

# Third panel: the spectrogram
    ax = plt.subplot(313)
    ax.imshow(wPSD, origin='lower', aspect='auto', cmap=plt.cm.jet,
          extent=[t[0], t[-1], f0[0], f0[-1]])

    ax.text(0.02, 0.95, ("Wavelet PSD"), color='w',
        ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(-4, 4)
    ax.set_ylim(0.5, 7.5)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$f_0$')

    plt.show()


def compute_Kernel():


    from scipy import optimize, fftpack, interpolate
    from astroML.fourier import IFT_continuous
    from astroML.filters import wiener_filter

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

#----------------------------------------------------------------------
# sample the same data as the previous Wiener filter figure
    np.random.seed(5)
    t = np.linspace(0, 100, 2001)[:-1]
    h = np.exp(-0.5 * ((t - 20.) / 1.0) ** 2)
    hN = h + np.random.normal(0, 0.5, size=h.shape)

#----------------------------------------------------------------------
# compute the PSD
    N = len(t)
    Df = 1. / N / (t[1] - t[0])
    f = fftpack.ifftshift(Df * (np.arange(N) - N / 2))

    h_wiener, PSD, P_S, P_N, Phi = wiener_filter(t, hN, return_PSDs=True)

#------------------------------------------------------------
# inverse fourier transform Phi to find the effective kernel
    t_plot, kernel = IFT_continuous(f, Phi)

#------------------------------------------------------------
# perform kernel smoothing on the data.  This is faster in frequency
# space (ie using the standard Wiener filter above) but we will do
# it in the slow & simple way here to demonstrate the equivalence
# explicitly
    kernel_func = interpolate.interp1d(t_plot, kernel.real)

    t_eval = np.linspace(0, 90, 1000)
    t_KDE = t_eval[:, np.newaxis] - t
    t_KDE[t_KDE < t_plot[0]] = t_plot[0]
    t_KDE[t_KDE > t_plot[-1]] = t_plot[-1]
    F = kernel_func(t_KDE)

    h_smooth = np.dot(F, hN) / np.sum(F, 1)

#------------------------------------------------------------
# Plot the results
    fig = plt.figure(figsize=(5, 2.2))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25,
                    bottom=0.15, top=0.9)

# First plot: the equivalent Kernel to the WF
    ax = fig.add_subplot(121)
    ax.plot(t_plot, kernel.real, '-k')
    ax.text(0.95, 0.95, "Effective Wiener\nFilter Kernel",
        ha='right', va='top', transform=ax.transAxes)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 0.45)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$K(\lambda)$')

# Second axes: Kernel smoothed results
    ax = fig.add_subplot(122)
    ax.plot(t_eval, h_smooth, '-k', lw=1)
    ax.plot(t_eval, 0 * t_eval, '-k', lw=1)
    ax.text(0.95, 0.95, "Kernel smoothing\nresult",
        ha='right', va='top', transform=ax.transAxes)

    ax.set_xlim(0, 90)
    ax.set_ylim(-0.5, 1.5)

    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('flux')

    plt.show()
  

def compute_AF():


    from astroML.time_series import lomb_scargle, generate_damped_RW
    from astroML.time_series import ACF_scargle, ACF_EK

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Generate time-series data:
#  we'll do 1000 days worth of magnitudes

    t = np.arange(0, 1E3)
    z = 2.0
    tau = 300
    tau_obs = tau / (1. + z)

    np.random.seed(6)
    y = generate_damped_RW(t, tau=tau, z=z, xmean=20)

# randomly sample 100 of these
    ind = np.arange(len(t))
    np.random.shuffle(ind)
    ind = ind[:100]
    ind.sort()
    t = t[ind]
    y = y[ind]

# add errors
    dy = 0.1
    y_obs = np.random.normal(y, dy)

#------------------------------------------------------------
# compute ACF via scargle method
    C_S, t_S = ACF_scargle(t, y_obs, dy,
                       n_omega=2 ** 12, omega_max=np.pi / 5.0)

    ind = (t_S >= 0) & (t_S <= 500)
    t_S = t_S[ind]
    C_S = C_S[ind]

#------------------------------------------------------------
# compute ACF via E-K method
    C_EK, C_EK_err, bins = ACF_EK(t, y_obs, dy, bins=np.linspace(0, 500, 51))
    t_EK = 0.5 * (bins[1:] + bins[:-1])

#------------------------------------------------------------
# Plot the results
    fig = plt.figure(figsize=(5, 5))

# plot the input data
    ax = fig.add_subplot(211)
    ax.errorbar(t, y_obs, dy, fmt='.k', lw=1)
    ax.set_xlabel('t (days)')
    ax.set_ylabel('observed flux')

# plot the ACF
    ax = fig.add_subplot(212)
    ax.plot(t_S, C_S, '-', c='gray', lw=1,
        label='Scargle')
    ax.errorbar(t_EK, C_EK, C_EK_err, fmt='.k', lw=1,
            label='Edelson-Krolik')
    ax.plot(t_S, np.exp(-abs(t_S) / tau_obs), '-k', label='True')
    ax.legend(loc=3)

    ax.plot(t_S, 0 * t_S, ':', lw=1, c='gray')

    ax.set_xlim(0, 500)
    ax.set_ylim(-1.0, 1.1)

    ax.set_xlabel('t (days)')
    ax.set_ylabel('ACF(t)')

    plt.show()
def compute_Wavelet():

    from astroML.fourier import FT_continuous, IFT_continuous

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)


    def wavelet(t, t0, f0, Q):
        return (np.exp(-(f0 / Q * (t - t0)) ** 2)
            * np.exp(2j * np.pi * f0 * (t - t0)))


    def wavelet_FT(f, t0, f0, Q):
    # this is its fourier transform using
    # H(f) = integral[ h(t) exp(-2pi i f t) dt]
        return (np.sqrt(np.pi) * Q / f0
            * np.exp(-2j * np.pi * f * t0)
            * np.exp(-(np.pi * (f - f0) * Q / f0) ** 2))


    def check_funcs(t0=1, f0=2, Q=3):
        t = np.linspace(-5, 5, 10000)
        h = wavelet(t, t0, f0, Q)

        f, H = FT_continuous(t, h)
        assert np.allclose(H, wavelet_FT(f, t0, f0, Q))

#------------------------------------------------------------
# Create the simulated dataset
    np.random.seed(5)

    t = np.linspace(-40, 40, 2001)[:-1]
    h = np.exp(-0.5 * ((t - 20.) / 1.0) ** 2)
    hN = h + np.random.normal(0, 0.5, size=h.shape)

#------------------------------------------------------------
# Compute the convolution via the continuous Fourier transform
# This is more exact than using the discrete transform, because
# we have an analytic expression for the FT of the wavelet.
    Q = 0.3
    f0 = 2 ** np.linspace(-3, -1, 100)

    f, H = FT_continuous(t, hN)
    W = np.conj(wavelet_FT(f, 0, f0[:, None], Q))
    t, HW = IFT_continuous(f, H * W)

#------------------------------------------------------------
# Plot the results
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0.05, left=0.12, right=0.95, bottom=0.08, top=0.95)

# First panel: the signal
    ax = fig.add_subplot(311)
    ax.plot(t, hN, '-k', lw=1)

    ax.text(0.02, 0.95, ("Input Signal:\n"
                     "Localized spike plus noise"),
        ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(-40, 40)
    ax.set_ylim(-1.2, 2.2)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylabel('$h(t)$')

# Second panel: the wavelet
    ax = fig.add_subplot(312)
    W = wavelet(t, 0, 0.125, Q)
    ax.plot(t, W.real, '-k', label='real part', lw=1)
    ax.plot(t, W.imag, '--k', label='imag part', lw=1)

    ax.legend(loc=1)
    ax.text(0.02, 0.95, ("Example Wavelet\n"
                     "$t_0 = 0$, $f_0=1/8$, $Q=0.3$"),
        ha='left', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.05,
        (r"$w(t; t_0, f_0, Q) = e^{-[f_0 (t - t_0) / Q]^2}"
         "e^{2 \pi i f_0 (t - t_0)}$"),
        ha='right', va='bottom', transform=ax.transAxes)

    ax.set_xlim(-40, 40)
    ax.set_ylim(-1.4, 1.4)
    ax.set_ylabel('$w(t; t_0, f_0, Q)$')
    ax.xaxis.set_major_formatter(plt.NullFormatter())

# Third panel: the spectrogram
    ax = fig.add_subplot(313)
    ax.imshow(abs(HW) ** 2, origin='lower', aspect='auto', cmap=plt.cm.binary,
          extent=[t[0], t[-1], np.log2(f0)[0], np.log2(f0)[-1]])
    ax.set_xlim(-40, 40)

    ax.text(0.02, 0.95, ("Wavelet PSD"), color='w',
        ha='left', va='top', transform=ax.transAxes)

    ax.set_ylim(np.log2(f0)[0], np.log2(f0)[-1])
    ax.set_xlabel('$t$')
    ax.set_ylabel('$f_0$')

    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, *args: ("1/%i"
                                                                 % (2 ** -x))))
    plt.show()


def compute_Sampling():


    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Generate the data
    Nbins = 2 ** 15
    Nobs = 40
    f = lambda t: np.sin(np.pi * t / 3)

    t = np.linspace(-100, 200, Nbins)
    dt = t[1] - t[0]
    y = f(t)

# select observations
    np.random.seed(42)
    t_obs = 100 * np.random.random(40)

    D = abs(t_obs[:, np.newaxis] - t)
    i = np.argmin(D, 1)

    t_obs = t[i]
    y_obs = y[i]
    window = np.zeros(Nbins)
    window[i] = 1

#------------------------------------------------------------
# Compute PSDs
    Nfreq = Nbins / 2

    dt = t[1] - t[0]
    df = 1. / (Nbins * dt)
    f = df * np.arange(Nfreq)

    PSD_window = abs(np.fft.fft(window)[:Nfreq]) ** 2
    PSD_y = abs(np.fft.fft(y)[:Nfreq]) ** 2
    PSD_obs = abs(np.fft.fft(y * window)[:Nfreq]) ** 2

# normalize the true PSD so it can be shown in the plot:
# in theory it's a delta function, so normalization is
# arbitrary

# scale PSDs for plotting
    PSD_window /= 500
    PSD_y /= PSD_y.max()
    PSD_obs /= 500

#------------------------------------------------------------
# Prepare the figures
    fig = plt.figure(figsize=(5, 2.5))
    fig.subplots_adjust(bottom=0.15, hspace=0.2, wspace=0.25,
                    left=0.12, right=0.95)

# First panel: data vs time
    ax = fig.add_subplot(221)
    ax.plot(t, y, '-', c='gray')
    ax.plot(t_obs, y_obs, '.k', ms=4)
    ax.text(0.95, 0.93, "Data", ha='right', va='top', transform=ax.transAxes)
    ax.set_ylabel('$y(t)$')
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.5, 1.8)

# Second panel: PSD of data
    ax = fig.add_subplot(222)
    ax.fill(f, PSD_y, fc='gray', ec='gray')
    ax.plot(f, PSD_obs, '-', c='black')
    ax.text(0.95, 0.93, "Data PSD", ha='right', va='top', transform=ax.transAxes)
    ax.set_ylabel('$P(f)$')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.1, 1.1)

# Third panel: window vs time
    ax = fig.add_subplot(223)
    ax.plot(t, window, '-', c='black')
    ax.text(0.95, 0.93, "Window", ha='right', va='top', transform=ax.transAxes)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y(t)$')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.2, 1.5)

# Fourth panel: PSD of window
    ax = fig.add_subplot(224)
    ax.plot(f, PSD_window, '-', c='black')
    ax.text(0.95, 0.93, "Window PSD", ha='right', va='top', transform=ax.transAxes)
    ax.set_xlabel('$f$')
    ax.set_ylabel('$P(f)$')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.1, 1.1)

    plt.show()

def compute_Power():

    from astroML.time_series import generate_power_law
    from astroML.fourier import PSD_continuous

    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

    N = 1024
    dt = 0.01
    factor = 100

    t = dt * np.arange(N)
    random_state = np.random.RandomState(1)

    fig = plt.figure(figsize=(5, 3.75))
    fig.subplots_adjust(wspace=0.05)

    for i, beta in enumerate([1.0, 2.0]):
    # Generate the light curve and compute the PSD
        x = factor * generate_power_law(N, dt, beta, random_state=random_state)
        f, PSD = PSD_continuous(t, x)

    # First axes: plot the time series
        ax1 = fig.add_subplot(221 + i)
        ax1.plot(t, x, '-k')

        ax1.text(0.95, 0.05, r"$P(f) \propto f^{-%i}$" % beta,
             ha='right', va='bottom', transform=ax1.transAxes)

        ax1.set_xlim(0, 10.24)
        ax1.set_ylim(-1.5, 1.5)

        ax1.set_xlabel(r'$t$')

    # Second axes: plot the PSD
        ax2 = fig.add_subplot(223 + i, xscale='log', yscale='log')
        ax2.plot(f, PSD, '-k')
        ax2.plot(f[1:], (factor * dt) ** 2 * (2 * np.pi * f[1:]) ** -beta, '--k')

        ax2.set_xlim(1E-1, 60)
        ax2.set_ylim(1E-6, 1E1)

        ax2.set_xlabel(r'$f$')

        if i == 1:
            ax1.yaxis.set_major_formatter(plt.NullFormatter())
            ax2.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax1.set_ylabel(r'${\rm counts}$')
            ax2.set_ylabel(r'$PSD(f)$')

    plt.show()

def compute_Chirp():

    from astroML.fourier import FT_continuous, IFT_continuous, wavelet_PSD

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)


#------------------------------------------------------------
# Define the chirp signal
    def chirp(t, T, A, phi, omega, beta):
        signal = A * np.sin(phi + omega * (t - T) + beta * (t - T) ** 2)
        signal[t < T] = 0
        return signal


    def background(t, b0, b1, Omega1, Omega2):
        return b0 + b1 * np.sin(Omega1 * t) * np.sin(Omega2 * t)

    N = 4096
    t = np.linspace(-50, 50, N)
    h_true = chirp(t, -20, 0.8, 0, 0.2, 0.02)
    h = h_true + np.random.normal(0, 1, N)

#------------------------------------------------------------
# Compute the wavelet PSD
    f0 = np.linspace(0.04, 0.6, 100)
    wPSD = wavelet_PSD(t, h, f0, Q=1.0)

#------------------------------------------------------------
# Plot the  results
    fig = plt.figure(figsize=(5, 3.75))
    fig.subplots_adjust(hspace=0.05, left=0.1, right=0.95, bottom=0.1, top=0.95)

# Top: plot the data
    ax = fig.add_subplot(211)
    ax.plot(t + 50, h, '-', c='#AAAAAA')
    ax.plot(t + 50, h_true, '-k')

    ax.text(0.02, 0.95, "Input Signal: chirp",
        ha='left', va='top', transform=ax.transAxes,
        bbox=dict(boxstyle='round', fc='w', ec='k'))

    ax.set_xlim(0, 100)
    ax.set_ylim(-2.9, 2.9)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylabel('$h(t)$')

# Bottom: plot the 2D PSD
    ax = fig.add_subplot(212)
    ax.imshow(wPSD, origin='lower', aspect='auto',
          extent=[t[0] + 50, t[-1] + 50, f0[0], f0[-1]],
          cmap=plt.cm.binary)

    ax.text(0.02, 0.95, ("Wavelet PSD"), color='w',
        ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(0, 100)
    ax.set_ylim(0.04, 0.6001)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$f_0$')

    plt.show()

def compute_Weiner():


    from scipy import optimize, fftpack
    from astroML.filters import savitzky_golay, wiener_filter

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Create the noisy data
    np.random.seed(5)
    N = 2000
    dt = 0.05

    t = dt * np.arange(N)
    h = np.exp(-0.5 * ((t - 20.) / 1.0) ** 2)
    hN = h + np.random.normal(0, 0.5, size=h.shape)

    Df = 1. / N / dt
    f = fftpack.ifftshift(Df * (np.arange(N) - N / 2))
    HN = fftpack.fft(hN)

#------------------------------------------------------------
# Set up the Wiener filter:
#  fit a model to the PSD consisting of the sum of a
#  gaussian and white noise
    h_smooth, PSD, P_S, P_N, Phi = wiener_filter(t, hN, return_PSDs=True)

#------------------------------------------------------------
# Use the Savitzky-Golay filter to filter the values
    h_sg = savitzky_golay(hN, window_size=201, order=4, use_fft=False)

#------------------------------------------------------------
# Plot the results
    N = len(t)
    Df = 1. / N / (t[1] - t[0])
    f = fftpack.ifftshift(Df * (np.arange(N) - N / 2))
    HN = fftpack.fft(hN)

    fig = plt.figure(figsize=(5, 3.75))
    fig.subplots_adjust(wspace=0.05, hspace=0.25,
                    bottom=0.1, top=0.95,
                    left=0.12, right=0.95)

# First plot: noisy signal
    ax = fig.add_subplot(221)
    ax.plot(t, hN, '-', c='gray')
    ax.plot(t, np.zeros_like(t), ':k')
    ax.text(0.98, 0.95, "Input Signal", ha='right', va='top',
        transform=ax.transAxes, bbox=dict(fc='w', ec='none'))

    ax.set_xlim(0, 90)
    ax.set_ylim(-0.5, 1.5)

    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('flux')

# Second plot: filtered signal
    ax = plt.subplot(222)
    ax.plot(t, np.zeros_like(t), ':k', lw=1)
    ax.plot(t, h_smooth, '-k', lw=1.5, label='Wiener')
    ax.plot(t, h_sg, '-', c='gray', lw=1, label='Savitzky-Golay')

    ax.text(0.98, 0.95, "Filtered Signal", ha='right', va='top',
        transform=ax.transAxes)
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.9), frameon=False)

    ax.set_xlim(0, 90)
    ax.set_ylim(-0.5, 1.5)

    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.set_xlabel(r'$\lambda$')

# Third plot: Input PSD
    ax = fig.add_subplot(223)
    ax.scatter(f[:N / 2], PSD[:N / 2], s=9, c='k', lw=0)
    ax.plot(f[:N / 2], P_S[:N / 2], '-k')
    ax.plot(f[:N / 2], P_N[:N / 2], '-k')

    ax.text(0.98, 0.95, "Input PSD", ha='right', va='top',
        transform=ax.transAxes)

    ax.set_ylim(-100, 3500)
    ax.set_xlim(0, 0.9)

    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel('$f$')
    ax.set_ylabel('$PSD(f)$')

# Fourth plot: Filtered PSD
    ax = fig.add_subplot(224)
    filtered_PSD = (Phi * abs(HN)) ** 2
    ax.scatter(f[:N / 2], filtered_PSD[:N / 2], s=9, c='k', lw=0)

    ax.text(0.98, 0.95, "Filtered PSD", ha='right', va='top',
        transform=ax.transAxes)

    ax.set_ylim(-100, 3500)
    ax.set_xlim(0, 0.9)

    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel('$f$')

    plt.show()    