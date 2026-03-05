

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt


def main():
    # sampling frequency
    ecg_timeseries_full=[]
    with open('ecg.csv') as f:
        for line in f.readlines():
            ecg_timeseries_full.append(float(line.strip()))
    # from the hardware specs used for recording this data, we know that
    # the sampling frequency is 500Hz
    sampling_freq = 500
    plt.figure()
    plt.title('Entire ECG signal')
    plt.plot(ecg_timeseries_full)
    plt.show()

    # the time-series is too long, so we're going to take a small segment
    x = ecg_timeseries_full[71740:81060]
    plt.figure()
    plt.plot(range(len(x)), x)
    plt.title('Cropped ECG signal')
    plt.show()

    freq1, pow1 = psd(x, sampling_freq)
    # Remember, x is the same data created earlier, covering samples 71,740 to 81,060
    plt.figure()
    plt.plot(freq1,pow1)
    plt.ylim([0, 0.005])
    plt.title('Periodogram')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(freq1,10*np.log10(pow1))
    plt.title('Periodogram Log scale')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.grid()
    plt.show()

def psd(input_signal, sampling_freq):

    # calculating the DFT using the FFT algorithm
    dft1 = np.fft.fft(input_signal)

    # getting rid of the second half of the DFT 
    # just a duplicate of the  first half
    dft1 = dft1[1:(len(input_signal)//2)+1]

    pow1 = (1/(sampling_freq*len(input_signal))) * abs(dft1)**2
    freq1 = np.linspace(0, sampling_freq//2, len(input_signal)//2)
    
    return freq1, pow1

def pwelch(x, fs=1.0, window='hamming', 
        nperseg=None, 
        noverlap=None, 
        nfft=None, 
        return_onesided=True, 
        scaling='density',
        average='mean'):
    
    """
    Estimate power spectral density using Welch's method.

    Parameters:
    x : array_like
        Time series of measurement values.
    fs : float, optional
        Sampling frequency of the x time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. Defaults to 'hamming'.
    nperseg : int, optional
        Length of each segment. If None, defaults to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If None, defaults to 50% overlap.
    nfft : int, optional
        Length of the FFT used. If None, the FFT length is nperseg.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. Defaults to True.
    scaling : {'density', 'spectrum'}, optional
        Selects between computing the power spectral density ('density') and the power spectrum ('spectrum'). Defaults to 'density'.
    average : {'mean', 'median'}, optional
        Method to use when averaging periodograms. Defaults to 'mean'.

    Returns:
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of x.
    """
    # Determine nperseg if not provided
    if nperseg is None:
        if isinstance(window, (str, tuple)):
            nperseg = 256
        else:
            nperseg = len(window)

    # Determine noverlap if not provided
    if noverlap is None:
        noverlap = nperseg // 2

    # Compute the PSD using welch function
    f, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                   return_onesided=return_onesided, scaling=scaling, average=average)

    return f, Pxx

if __name__ == "__main__":
    main()
