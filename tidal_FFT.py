# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:34:13 2022

@author: toru1
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import fft

# define the frequencies of the M2, S2, and M4 harmonics
HARMONICS = {
     'M4': 6.21030061,
     'S2': 12.00000000,
     'M2': 12.42060121,
     'N2': 12.65834824,
    }


# %%
def FFT_tide(data):
    header = list(data)[0]
    WL = data[header].to_numpy()
    # mask = ~np.isnan(WL)
    # WL = WL[mask]
    MeanWL = np.nanmean(WL)
    print('Mean WL [cm]: {:.2f}'.format(MeanWL))

    # Frequency and sampling rate
    t = data.index.to_numpy()
    print('Date: {} - {}'.format(t[0], t[-1]))
    # t = t[mask]
    n = t.size
    # Perform Fourier transform using scipy
    fft_data = fft.rfft(WL)[1:]           # fft = n/2*amplitude
    Amplitude = 2/n * np.abs(fft_data)
    Phase = np.angle(fft_data)
    timestep = 1/60             # hour/min
    fr = fft.rfftfreq(n, d=timestep)[1:]

    fft_data_2 = fft_data*0

    print("NAME\t Amplitude [cm]\t Phase \tIndex")

    Harmonic = pd.DataFrame({}, index=HARMONICS.keys(), 
                            columns=["Amplitude [cm]", "Phase", "Index"])
    for key in HARMONICS:
        # search for the M2, S2, and M4 frequencies in the FFT data
        index = np.argmin(np.abs(fr - 1/HARMONICS[key]))

        # print the results
        print('{} \t{:.2f} \t{:.2f} \t{}'
              .format(key, Amplitude[index], Phase[index], index))

        # store selected harmonic data
        fft_data_2[index] = fft_data[index]
        Harmonic.at[key, "Amplitude [cm]"] = Amplitude[index]
        Harmonic.at[key, "Phase"] = Phase[index]
        Harmonic.at[key, "Index"] = index

    # inverse of FFT to check tide
    ifft = fft.irfft(fft_data_2)
    Amplitude2 = 2/n * np.abs(fft_data_2)
    Time = 1 / fr
    return WL, Amplitude, Phase, Time, Amplitude2, ifft, Harmonic


# %%Plot figures
def Plot_FFT_Tide(WL, Amplitude, Phase, Time, Amplitude2, ifft, Harmonic):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    ax.plot(WL)        # plot tidal data
    ax.set_ylabel('Water level [cm]')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.plot(Time, Amplitude)        # plot freq domain
    flag=0
    for key in HARMONICS:
        ax.axvline(HARMONICS[key], color='grey', linestyle='--', linewidth=0.5)
        ax.text(HARMONICS[key], (flag%3)*50-5, key, color='grey', horizontalalignment='center')
        flag += 1
    ax.set_xlim(0, 36)              # hour
    ax.set_ylabel('Amplitude [cm]')
    ax.set_xlabel('Time [h]')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    # ax.plot(fr, np.angle(fft_data))        # plot phase
    # ax.set_xlim(0, 1)
    # plt.savefig('FFT_Osteriff_WL.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    ax.plot(Time, Amplitude2)
    ax.set_xlim(0, 36)              # hour

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    ax.plot(ifft)
    ax.set_xlim(0, 1000000)


def FindHeader(file, marker):
    # find header (after line of marker)
    skipcol = 0
    with open(file) as f:
        while True:
            line = f.readline()
            if line[0] == marker:
                break
            skipcol += 1
    return skipcol


def SplitByNan(df, threshold):
    # split time-series by NAN
    print('Splitting data...')
    marker = np.array(np.where(df.isna().any(axis=1))[0])
    flag1, flag2 = [], []
    isna = np.array(df.isna().any(axis=1))
    for i in range(1, len(isna)-2):
        # find start of Nan position
        if isna[i] and isna[i+1] and not isna[i-1]:
            flag1.append(i)
        # find end of Nan position
        elif isna[i] and isna[i-1] and not isna[i+1]:
            flag2.append(i+1)
    flag1, flag2 = np.array(flag1), np.array(flag2)
    diff = flag2 - flag1
    # ignore small nan entries
    ignoreNan = 60
    marker = np.unique(np.concatenate([flag1[diff > ignoreNan],
                                       flag2[diff > ignoreNan]]))
    data = np.split(df, marker)
#    nans = np.array([d.isna().sum().sum() for d in data2])
    # removing small entries
    data = [d for i, d in enumerate(data) if i % 2 == 0 and d.size > threshold]
    data2 = [d.interpolate() for d in data]
    return data2


# %% reading data
#file = 'C:/Users/toru1/OneDrive - KU Leuven/Thesis/3_Schematized_elbe_' \
#        'calibration/Toru_15122022/bake_c!Wasserstand_(NN-Bezug).txt'
file = input('Paste absolute path of tidal data \n>>> ')
print("Reading file...")

header = FindHeader(file, '=') + 1
df = pd.read_table(file, header=header, index_col=0, usecols=[0, 1])[:-1]
data = SplitByNan(df, threshold=60*24*30)

# %% main
FFT_data = []
Harmonic_data = []
IFFT = []
# Calculate FFT
for i, d in enumerate(data):
#    if i == 6:
        print("\nProcessing...{}/{}".format(i+1, len(data)))
        WL, Amplitude, Phase, Time, Amplitude2, ifft, Harmonic = FFT_tide(d)
        FFT_data.append(np.array([Amplitude, Phase, Time, Amplitude2]))
        Harmonic_data.append(Harmonic)
        IFFT.append(ifft)
    
        print("Plotting...{}/{}".format(i+1, len(data)))
        Plot_FFT_Tide(WL, Amplitude, Phase, Time, Amplitude2, ifft, Harmonic)

if len(Harmonic_data)!=0:
    Harmonic_data = pd.concat(Harmonic_data)

# print number of data
# for d in data:
#     print(d.size)

# %% main 2
r"""
path = r'C:\Users\toru1\OneDrive - KU Leuven\Thesis' \
   '\3_Schematized_elbe_calibration\tidal data'
file = os.path.join(path, r'Osteriff_WL.csv')

file = input('Paste absolute path of tidal data \n>>> ')
data = pd.read_csv(file, parse_dates=True)
data = data.interpolate()
Amplitude, Phase, Time, Amplitude2, ifft = FFT_tide(data)

"""
