import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz


# parameters

neig = 2	# number eigenvalues for reconstruction
fs = 4096.	# sampling frequency
f0 = 64.	# signal frequency
N0 = 1024	# number of samples
SNR = -10.	# SNR in dB

# signal generation

noiseStd = np.sqrt(1/(2*10**(SNR/10)))		# noise std
noise = noiseStd*np.random.randn(N0)		# noise
SOI = np.sin(2.*np.pi*f0/fs*np.arange(N0))	# signal of interest = SINEWAVE
# SOI = np.sin(2.*np.pi*(0.1 + 0.1/2./N0*np.arange(N0))*np.arange(N0))	# signal of interest = CHIRP
SigReceived =  SOI + noise					# received signal

# auto correlation matrix

CCF_array = signal.correlate(SigReceived, SigReceived, mode='full')
Nmax = np.argmax(abs(CCF_array))		# max index
ACM = toeplitz(CCF_array[Nmax:Nmax+N0])	# AutoCorrelation Matrix
   
# KLT

EVal,EVec = np.linalg.eigh(ACM)
    
if EVal[0] < EVal[-1]:	# change to decreasing order if needed
    EVal = np.flipud(EVal)
    EVec = np.fliplr(EVec)

EVec = EVec[:,0:neig]	# limit dimensions of reconstruction
projVec = np.dot(np.transpose(EVec), SigReceived)
RecSignal = np.dot(EVec,projVec)


plt.figure()
plt.subplot(2,2,1)
plt.title('SNR = '+str(SNR)+'dB - ' + str(neig) + ' value(s) reconstruction')
plt.plot(SigReceived,color='b',label='received signal')
plt.plot(RecSignal,color='r',label='reconstructed signal')
plt.xlim((0,N0))
plt.legend(loc=0)

plt.subplot(2,2,2)
plt.title('Comparison original / reconstructed signals')
plt.plot(SOI,color='b',label='original signal')
plt.plot(RecSignal,color='r',label='reconstructed signal')
plt.legend(loc=0)
plt.xlim((0,N0))

plt.subplot(2,2,3)
plt.title('ACM eigenvalues')
plt.plot(range(neig),EVal[:neig],'o',color='r',label='chosen eigenvalues')
plt.plot(range(neig,N0),EVal[neig:],'o',color='b',label='other eigenvalues')
plt.xlim((-1,20))
plt.legend(loc=0)

fft_SOI = np.fft.fft(SOI)
fft_reconstructed = np.fft.fft(RecSignal)
plt.subplot(2,2,4)
plt.title('Comparison FFT original / reconstructed signals')
plt.plot(np.arange(0.,1.,2./float(N0)),10.*np.log10(np.abs(fft_SOI[0:np.size(fft_SOI)/2])**2),color='b',label='original signal')
plt.plot(np.arange(0.,1.,2./float(N0)),10.*np.log10(np.abs(fft_reconstructed[0:np.size(fft_reconstructed)/2])**2),color='r',label='reconstructed signal')
plt.legend(loc=0)
plt.xlim((0,1))


plt.show()