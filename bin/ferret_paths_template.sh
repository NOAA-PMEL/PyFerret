##
## Environment settings for PyFerret for Bourne shell users. 
## Source this file before running pyferret ('. ferret_paths.sh')
##

## The environment variable FER_DIR should be the pathname of the directory 
## of the PyFerret software.
export FER_DIR="##FER_DIR##"

## The environment variable FER_DSETS should be the pathname of the directory 
## of the Ferret demonstration data files.
export FER_DSETS="##FER_DSETS##"

## Set python_exe to the (optionally full-path) python executable to use
python_exe="##PYTHON_EXECUTABLE##"

## Directory containing the pyferret Python module subdirectory
## Typically this is "${FER_DIR}/lib/pythonX.x/site-packages" 
## where X and x are the Python major and minor version numbers.
pysite="##PYFERRET_SITE##"

## =========== The remainder of this file should not need modification ===========
## =========== unless you want to add custom directories or sites to   ===========
## =========== the Ferret's defaults.                                  ===========


## Add ${FER_DIR}/bin to the beginning of ${PATH}
if ! echo "${PATH}" | grep -q "^${FER_DIR}/bin:"; then
    export PATH="${FER_DIR}/bin:${PATH}"
fi

## For Mac OS X, add ${FER_DIR}/dylibs to DYLD_FALLBACK_LIBRARY_PATH
if [ ! -z "$DYLD_FALLBACK_LIBRARY_PATH" ]; then
    if ! echo "${DYLD_FALLBACK_LIBRARY_PATH}" | grep -q "${FER_DIR}/dylibs"; then
        export DYLD_FALLBACK_LIBRARY_PATH="${FER_DIR}/dylibs:${DYLD_FALLBACK_LIBRARY_PATH}"
    fi
else
    export DYLD_FALLBACK_LIBRARY_PATH="${FER_DIR}/dylibs"
fi

## Space-separated lists of directories examined when searching for 
## data, descriptor, grid, and go-script files without path components.
export FER_DATA=". ${FER_DSETS}/data ${FER_DIR}/contrib"
export FER_DESCR=". ${FER_DSETS}/descr"
export FER_GRIDS=". ${FER_DSETS}/grids"
export FER_GO=". ${FER_DIR}/go ${FER_DIR}/examples ${FER_DIR}/contrib"

## Space-separated list of directories containing traditional
## PyFerret external function files (shared-object libraries)
export PYFER_EXTERNAL_FUNCTIONS="${FER_DIR}/ext_func/pylibs"

## Space-separated list of directories for Ferret color palettes
export FER_PALETTE=". ${FER_DIR}/ppl"

## Directory for Ferret fonts
export FER_FONTS="${FER_DIR}/ppl/fonts"

## Add $pysite to the Python search path given by PYTHONPATH 
## so the pyferret package will be found.
if [ -z "${PYTHONPATH}" ]; then
    export PYTHONPATH="${pysite}"
else
    if ! echo "${PYTHONPATH}" | grep -q "^${pysite}"; then
        export PYTHONPATH="${pysite}:${PYTHONPATH}"
    fi
fi

## Add $pysite/pyferret to the shared-object library search path given 
## by LD_LIBRARY_PATH so libpyferret.so will be found by the Fortran EFs.
if [ -z "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${pysite}/pyferret"
else
    if ! echo "${LD_LIBRARY_PATH}" | grep -q "^${pysite}/pyferret"; then
        export LD_LIBRARY_PATH="${pysite}/pyferret:${LD_LIBRARY_PATH}"
    fi
fi

## Faddpath: a tool to quickly add paths to the search lists
Faddpath() { if [ -n "$*" ]
             then
                 export FER_GO="$FER_GO $*"
                 export FER_DATA="$FER_DATA $*"
                 export FER_DESCR="$FER_DESCR $*"
                 export FER_GRIDS="$FER_GRIDS $*"
             else
                 echo "    Usage: Faddpath new_directory_1 ..."
             fi }

