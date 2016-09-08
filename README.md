# PyFerret
The PyFerret Python module from NOAA/PMEL.
This repository is regularly synchronized with PyFerret repository at PMEL
(the pyferret branch of the ferret project in the TMAP SVN repository at PMEL)
using git-svn.

## Anaconda Installation - Linux, OS X, and Windows 10/bash

Download and install [miniconda](http://conda.pydata.org/miniconda.html) for your system. Note that Windows 10 bash must use the Linux version! The Python version in miniconda does not matter; `pyferret` only uses ` Python 2.7` and will install it in the `pyferret` environment.

Execute the following command on the terminal to install `pyferret` into conda:
```shell
conda create -n FERRET -c conda-forge pyferret --yes
```

To start using `pyferret`, execute the following command:
```shell
source activate FERRET
```

Once you are done working with `pyferret` you can leave this environment, if you wish, with the command:
```shell
source deactivate FERRET
```

In the commands above, `FERRET` is the environment name where `pyferret` is installed. You can change that to any name you like but we do not recommend installing `pyferret` in the root environment of miniconda. The main reason is to take advantage of the `activate/deactivate` script that will set up all the variables that `pyferret` needs. (You can test whether the `pyferret` environment is activated by issuing the command `echo $FER_DATA` and see if it returns a directory name.)

