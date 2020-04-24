# Channel Estimation Challenge
The channel estimation challenge assumes a multiple-input multiple-output (MIMO) communication system using millimeter wave (mmWave) frequencies. In this challenge, the inputs are preamble (or pilot) signals obtained at the receiver, after propagation through the channel. The channels are obtained with ray-tracing simulations. Based on the received pilots, the task is to estimate the MIMO channel.

## Requirements
Requirements to run the baseline model are:

python >= 3.7.5

h5py >= 2.8.0

numpy >= 1.18.1

keras >= 2.3.1

scikit-learn >= 0.22

## How to run the code
Follow steps below to run the baseline models
### 1. download dateset
Inside dataset folder, run the following command in terminal.

```bash
python download_dataset.py
```

### 2. run main.py
You can run main.py scritp passing arguments in the terminal to tell what you want to do. You can train the model, or you can train and test or you can only test:

a) to train and test (default option): `python main.py`

b) just to train: `python main.py --mode train`    

c) just to test: `python main.py --mode test`

Before you run, please create a folder named `models` in the same directory of main.py file to save your models.




# Name                    Version                   Build  Channel
absl-py                   0.9.0                    pypi_0    pypi
alabaster                 0.7.12                   py37_0
asn1crypto                1.3.0                    py37_0
astor                     0.8.1                    pypi_0    pypi
astroid                   2.3.3                    py37_0
attrs                     19.3.0                     py_0
babel                     2.8.0                      py_0
backcall                  0.1.0                    py37_0
blas                      1.0                         mkl
bleach                    3.1.4                      py_0
ca-certificates           2020.1.1                      0
cachetools                4.0.0                    pypi_0    pypi
certifi                   2020.4.5.1               py37_0
cffi                      1.14.0           py37h7a1dbc1_0
chardet                   3.0.4                 py37_1003
cloudpickle               1.3.0                      py_0
colorama                  0.4.3                      py_0
cryptography              2.8              py37h7a1dbc1_0
cycler                    0.10.0                   py37_0
decorator                 4.4.2                      py_0
defusedxml                0.6.0                      py_0
docutils                  0.16                     py37_0
entrypoints               0.3                      py37_0
freetype                  2.9.1                ha9979f8_1
gast                      0.2.2                    pypi_0    pypi
google-auth               1.10.0                   pypi_0    pypi
google-auth-oauthlib      0.4.1                    pypi_0    pypi
google-pasta              0.1.8                    pypi_0    pypi
grpcio                    1.26.0                   pypi_0    pypi
h5py                      2.8.0            py37hf7173ca_2
hdf5                      1.8.20               hac2f561_1
icc_rt                    2019.0.0             h0cc432a_1
icu                       58.2                 ha66f8fd_1
idna                      2.8                      pypi_0    pypi
imagesize                 1.2.0                      py_0
importlib_metadata        1.5.0                    py37_0
intel-openmp              2020.0                      166
ipykernel                 5.1.4            py37h39e3cac_0
ipython                   7.13.0           py37h5ca1d4c_0
ipython_genutils          0.2.0                    py37_0
ipywidgets                7.5.1                      py_0
isort                     4.3.21                   py37_0
jedi                      0.16.0                   py37_1
jinja2                    2.11.1                     py_0
joblib                    0.14.1                     py_0
jpeg                      9b                   hb83a4c4_2
jsonschema                3.2.0                    py37_0
jupyter                   1.0.0                    py37_7
jupyter_client            6.1.2                      py_0
jupyter_console           6.1.0                      py_0
jupyter_core              4.6.3                    py37_0
keras                     2.3.1                    pypi_0    pypi
keras-applications        1.0.8                    pypi_0    pypi
keras-preprocessing       1.1.0                    pypi_0    pypi
keyring                   21.1.1                   py37_2
kiwisolver                1.1.0            py37ha925a31_0
lazy-object-proxy         1.4.3            py37he774522_0
libopencv                 3.4.2                h20b85fd_0
libpng                    1.6.37               h2a8f88b_0
libsodium                 1.0.16               h9d3ae62_0
libtiff                   4.1.0                h56a325e_0
m2w64-gcc-libgfortran     5.3.0                         6
m2w64-gcc-libs            5.3.0                         7
m2w64-gcc-libs-core       5.3.0                         7
m2w64-gmp                 6.1.0                         2
m2w64-libwinpthread-git   5.0.0.4634.697f757               2
markdown                  3.1.1                    pypi_0    pypi
markupsafe                1.1.1            py37he774522_0
matplotlib                3.1.1            py37hc8f65d3_0
mccabe                    0.6.1                    py37_1
mistune                   0.8.4            py37he774522_0
mkl                       2020.0                      166
mkl-service               2.3.0            py37hb782905_0
mkl_fft                   1.0.15           py37h14836fe_0
mkl_random                1.1.0            py37h675688f_0
msys2-conda-epoch         20160418                      1
nbconvert                 5.6.1                    py37_0
nbformat                  5.0.4                      py_0
notebook                  6.0.3                    py37_0
numpy                     1.18.1           py37h93ca92e_0
numpy-base                1.18.1           py37hc3f5095_1
numpydoc                  0.9.2                      py_0
oauthlib                  3.1.0                    pypi_0    pypi
opencv                    3.4.2            py37h40b0b35_0
openssl                   1.1.1g               he774522_0
opt-einsum                3.1.0                    pypi_0    pypi
packaging                 20.3                       py_0
pandas                    1.0.1            py37h47e9c7a_0
pandoc                    2.2.3.2                       0
pandocfilters             1.4.2                    py37_1
parso                     0.6.2                      py_0
patsy                     0.5.1                    py37_0
pickleshare               0.7.5                    py37_0
pip                       20.0.2                   py37_1
prometheus_client         0.7.1                      py_0
prompt-toolkit            3.0.4                      py_0
prompt_toolkit            3.0.4                         0
protobuf                  3.11.1                   pypi_0    pypi
psutil                    5.6.7            py37he774522_0
py-opencv                 3.4.2            py37hc319ecb_0
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.7                    pypi_0    pypi
pycodestyle               2.5.0                    py37_0
pycparser                 2.20                       py_0
pyflakes                  2.1.1                    py37_0
pygments                  2.6.1                      py_0
pylint                    2.4.4                    py37_0
pyopenssl                 19.1.0                   py37_0
pyparsing                 2.4.6                      py_0
pyqt                      5.9.2            py37h6538335_2
pyrsistent                0.16.0           py37he774522_0
pysocks                   1.7.1                    py37_0
python                    3.7.5                h8c8aaf0_0
python-dateutil           2.8.1                      py_0
pytz                      2019.3                     py_0
pywget                    3.2                      py37_0
pywin32                   227              py37he774522_1
pywin32-ctypes            0.2.0                 py37_1000
pywinpty                  0.5.7                    py37_0
pyyaml                    5.2                      pypi_0    pypi
pyzmq                     18.1.1           py37ha925a31_0
qt                        5.9.7            vc14h73c81de_0
qtawesome                 0.7.0                      py_0
qtconsole                 4.7.3                      py_0
qtpy                      1.9.0                      py_0
requests                  2.22.0                   pypi_0    pypi
requests-oauthlib         1.3.0                    pypi_0    pypi
rope                      0.16.0                     py_0
rsa                       4.0                      pypi_0    pypi
scikit-learn              0.22             py37h6288b17_0
scipy                     1.3.2            py37h29ff71c_0
send2trash                1.5.0                    py37_0
setuptools                46.1.3                   py37_0
sip                       4.19.8           py37h6538335_0
six                       1.14.0                   py37_0
snowballstemmer           2.0.0                      py_0
sphinx                    2.4.4                      py_0
sphinxcontrib-applehelp   1.0.2                      py_0
sphinxcontrib-devhelp     1.0.2                      py_0
sphinxcontrib-htmlhelp    1.0.3                      py_0
sphinxcontrib-jsmath      1.0.1                      py_0
sphinxcontrib-qthelp      1.0.3                      py_0
sphinxcontrib-serializinghtml 1.1.4                      py_0
spyder                    3.3.6                    py37_0
spyder-kernels            0.5.2                    py37_0
sqlite                    3.31.1               he774522_0
statsmodels               0.11.0           py37he774522_0
tensorboard               2.0.2                    pypi_0    pypi
tensorflow-estimator      2.0.1                    pypi_0    pypi
tensorflow-gpu            2.0.0                    pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
terminado                 0.8.3                    py37_0
testpath                  0.4.4                      py_0
tornado                   6.0.4            py37he774522_1
traitlets                 4.3.3                    py37_0
urllib3                   1.25.7                   pypi_0    pypi
vc                        14.1                 h0510ff6_4
vs2015_runtime            14.16.27012          hf0eaf9b_1
wcwidth                   0.1.9                      py_0
webencodings              0.5.1                    py37_1
werkzeug                  0.16.0                   pypi_0    pypi
wheel                     0.34.2                   py37_0
widgetsnbextension        3.5.1                    py37_0
win_inet_pton             1.1.0                    py37_0
wincertstore              0.2                      py37_0
winpty                    0.4.3                         4
wrapt                     1.12.1           py37he774522_1
xz                        5.2.5                h62dcd97_0
zeromq                    4.3.1                h33f27b4_3
zipp                      2.2.0                      py_0
zlib                      1.2.11               h62dcd97_4
zstd                      1.3.7                h508b16e_0