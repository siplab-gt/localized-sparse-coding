"""
Fit Gabor wavelet to trained dictionry. Required Matlab Python engine, Python 3.6 or 3.7, Curve Fitting Toolbox,
and Image Processing Toolbox for Matlab.

@Filename    fit_gabor.py
@Author      Kion 
@Created     5/31/20
"""
import argparse
import glob
import io
import os

import matlab.engine
import numpy as np

parser = argparse.ArgumentParser(description='Fit Gabor wavelet to trained dictionaries')
parser.add_argument('-s', '--sweep', action='store_true', help="Sweep all files in ./results")
parser.add_argument('-i', '--input', help="Specify input file")
parser.add_argument('-r', '--runs', default=100, type=int, help="Number of runs for each wavelet fit")
args = parser.parse_args()

if not args.sweep and args.input is None:
    raise ValueError("If not sweeping (--sweep) must specify input (--input <FILE>).")
elif not args.sweep:
    file_list = [args.input]
else:
    file_list = glob.glob("./results/*.npz")

if __name__ == "__main__":
    # Load Matlab engine
    print("Starting Matlab engine...")
    eng = matlab.engine.start_matlab()
    # Add utils to path
    eng.addpath(r'./utils/', nargout=0)

    # Options used by Matlab script
    options = {'shape': 'elliptical', 'runs': args.runs, 'parallel': False, 'visualize': False}

    for file_path in file_list:
        save_path = file_path.replace('traindata', 'wavelet_fit')
        if os.path.exists(save_path):
            print("Fit already found, skipping {}".format(file_path))
            continue

        print("Loading {}...".format(file_path))
        # Load .npz file
        data_file = np.load(file_path)
        # Extract dictionary list from loaded data file
        dictionary_list = data_file['phi']
        # Pick the dictionary from the last epoch
        dictionary = dictionary_list[-1]
        # Find dictionary shape and reshape
        patch_size = int(np.sqrt(dictionary.shape[0]))
        dictionary = dictionary.reshape(patch_size, patch_size, -1)

        # Loop through each atom in dictionary and find best wavelet fit
        a = []
        b = []
        x0 = []
        y0 = []
        sigmax = []
        sigmay = []
        theta = []
        phi = []
        Lambda = []
        phase = []
        for i in range(dictionary.shape[-1]):
            # try:
            atom_matlab = matlab.double(dictionary[:, :, i].tolist())
            wavelet_fit = eng.fit2dGabor(atom_matlab, options, stdout=io.StringIO())['fit']
            # TODO: Improve code quality
            a.append(wavelet_fit['a'])
            b.append(wavelet_fit['b'])
            x0.append(wavelet_fit['x0'])
            y0.append(wavelet_fit['y0'])
            sigmax.append(wavelet_fit['sigmax'])
            sigmay.append(wavelet_fit['sigmay'])
            theta.append(wavelet_fit['theta'])
            phi.append(wavelet_fit['phi'])
            Lambda.append(wavelet_fit['lambda'])
            phase.append(wavelet_fit['phase'])
            print("SUCCESS to fit wavelet to dictionary element {} of {}".format(i + 1, dictionary.shape[-1]))
            # except:
            #     print("FAILED to fit wavelet to dictionary element {} of {}".format(i + 1, dictionary.shape[-1]))
            #     continue

        np.savez_compressed(save_path, a=np.array(a), b=np.array(b), x0=np.array(x0), y0=np.array(y0),
                            sigmax=np.array(sigmax), sigmay=np.array(sigmay), theta=np.array(theta),
                            phi=np.array(phi), Lambda=np.array(Lambda), phase=np.array(phase))
        print("... Saved in {}".format(save_path))
