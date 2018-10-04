# -*- coding: utf-8 -*-
"""
example of colored noise generation
"""

import numpy as np
import matplotlib.pyplot as plt

nSam = 1024 # number of samples

noise = np.random.normal(0.,1.,nSam)

plt.figure()
plt.subplot(211)
plt.title('white noise')
plt.plot(noise)
plt.xlim(0,nSam)
plt.xlabel('time samples')
plt.ylabel('amplitude')
plt.subplot(212)
plt.title('noise spectrum')
plt.plot(np.fft.fftshift(np.fft.fftfreq(noise.shape[-1])),10.*np.log10(np.fft.fftshift(np.abs(np.fft.fft(noise))**2)))
plt.xlim(-0.5,0.5)
plt.xlabel('normalized frequencies')
plt.ylabel('power [dB]')

hanwin = np.hanning(32)
window = np.fft.fftshift(hanwin)    # window with attenuation at the edges
filt = np.fft.fftshift(np.fft.ifft(window))     # ifft of window to convolve in time domain

noise_colored = np.convolve(noise, filt, mode='same')

plt.figure()
plt.subplot(311)
plt.title('Hanning colored noise')
plt.plot(noise_colored)
plt.xlim(0,nSam)
plt.xlabel('time samples')
plt.ylabel('amplitude')
plt.subplot(312)
plt.title('noise spectrum')
plt.plot(np.fft.fftshift(np.fft.fftfreq(noise.shape[-1])),10.*np.log10(np.fft.fftshift(np.abs(np.fft.fft(noise_colored))**2)))
plt.xlim(-0.5,0.5)
plt.xlabel('normalized frequencies')
plt.ylabel('power [dB]')
plt.subplot(313)
plt.title('Hanning window')
plt.plot(hanwin)
plt.xlim(0,31)
plt.ylim(-0.1,1.1)



cuswin = np.ones(64);cuswin[0:8]=np.arange(0.,1.,1./8.);cuswin[-9:-1]=np.arange(1.,0.,-1./8.);cuswin[-1]=0.
window = np.fft.fftshift(cuswin)    # window with attenuation at the edges
filt = np.fft.fftshift(np.fft.ifft(window))     # ifft of window to convolve in time domain

noise_colored = np.convolve(noise, filt, mode='same')

plt.figure()
plt.subplot(311)
plt.title('Custom colored noise')
plt.plot(noise_colored)
plt.xlim(0,nSam)
plt.xlabel('time samples')
plt.ylabel('amplitude')
plt.subplot(312)
plt.title('noise spectrum')
plt.plot(np.fft.fftshift(np.fft.fftfreq(noise.shape[-1])),10.*np.log10(np.fft.fftshift(np.abs(np.fft.fft(noise_colored))**2)))
plt.xlim(-0.5,0.5)
plt.xlabel('normalized frequencies')
plt.ylabel('power [dB]')
plt.subplot(313)
plt.title('Custom window')
plt.plot(cuswin)
plt.xlim(0,63)
plt.ylim(-0.1,1.1)
