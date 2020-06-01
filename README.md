# Compressed Sparse Dictionary

### Background

< Have Chris give a brief background on the project>

### Usage
#### Training
In order to train a compressed sparse dictionary, use the `train_sparse_dict.py` script. There are several optional parameters, but the ones that you might be interested in are:
 
* `--compression` or `-c`, which determines the compression matrix. Options are `none` for no compression, `dbd` for distinct block diagonal, and `bd` for banded diagonal.
* `--localization` or `-j`, which determines the degree of localization

Example usage:
```
python train_script_dict.py -c dbd -j 4
```
#### Analysis
To analyse pre-trained dictionaries, use data_plotting.ipynb. This is a python notebook that requires Jupyter to interact with.

There is an additional script included to fit Gabor wavelets to all trained dictionaries for further analysis. This script requires Matlab, and is built off a script from [Gerrit Ecke](https://www.mathworks.com/matlabcentral/fileexchange/60700-fit2dgabor-data-options). It can be run with the following command:
```
python fit_gabor.py --sweep
```

#### Dependencies
Python 3.0+, Scikit-Learn, Numpy, Scipy. 

To use Matlab engine, Python 3.6 or 3.7 required with Matlab. [See this StackOverflow post for details](https://stackoverflow.com/questions/46141631/running-matlab-using-python-gives-no-module-named-matlab-engine-error). 

### Examples

#### Uncompressed
<p align="center">
<img align="middle" src="./results/uncompressed.gif" alt="Uncompressed" width="512" height="512" />
</p>

#### Distinct Block Diagonal Matrix, J = 4
<p align="center">
<img align="middle" src="./results/block_diagonal.gif" alt="DBD" width="512" height="512" />
</p>

#### Banded Diagonal Matrix, J = 4
<p align="center">
<img align="middle" src="./results/banded_diagonal.gif" alt="BD" width="512" height="512" />
</p>