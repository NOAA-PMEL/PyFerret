##
## Environment settings for PyFerret for C shell users. 
## Source this file before running pyferret ('source ferret_paths.csh')
##

## The environment variable FER_DIR should be the pathname of the directory 
## of the PyFerret software.
setenv FER_DIR "##FER_DIR##"

## The environment variable FER_DSETS should be the pathname of the directory 
## of the Ferret demonstration data files.
setenv FER_DSETS "##FER_DSETS##"

## Set python_exe to the (optionally full-path) python executable to use
set python_exe = "##PYTHON_EXECUTABLE##"

## Directory containing the pyferret Python module subdirectory
## Typically this is "${FER_DIR}/lib/pythonX.x/site-packages" 
## where X and x are the Python major and minor version numbers.
set pysite = "##PYFERRET_SITE##"


## =========== The remainder of this file should not need modification ===========
## =========== unless you want to add custom directories or sites to   ===========
## =========== the Ferret's defaults.                                  ===========


## Add ${FER_DIR}/bin to the beginning of ${PATH}
if ( "${PATH}" !~ "${FER_DIR}/bin:*" ) then
    setenv PATH "${FER_DIR}/bin:${PATH}"
    rehash
endif

## Space-separated lists of directories examined when searching for 
## data, descriptor, grid, and go-script files without path components.
setenv FER_DATA ". ${FER_DSETS}/data ${FER_DIR}/contrib"
setenv FER_DESCR ". ${FER_DSETS}/descr"
setenv FER_GRIDS ". ${FER_DSETS}/grids"
setenv FER_GO ". ${FER_DIR}/go ${FER_DIR}/examples ${FER_DIR}/contrib"

## Space-separated list of directories containing traditional
## PyFerret external function files (shared-object libraries)
setenv PYFER_EXTERNAL_FUNCTIONS "${FER_DIR}/ext_func/pylibs"

## Space-separated list of directories for Ferret color palettes
setenv FER_PALETTE ". ${FER_DIR}/ppl"

## Directory for Ferret fonts
setenv FER_FONTS "${FER_DIR}/ppl/fonts"

## Add $pysite to the Python search path given by PYTHONPATH 
## so the pyferret package will be found.
if ( ! $?PYTHONPATH ) then
    setenv PYTHONPATH "${pysite}"
else
    if ( "${PYTHONPATH}" !~ "${pysite}*" ) then
        setenv PYTHONPATH "${pysite}:${PYTHONPATH}"
    endif
endif

## Add $pysite/pyferret to the shared-object library search path given 
## by LD_LIBRARY_PATH so libpyferret.so will be found by the Fortran EFs.
if ( ! $?LD_LIBRARY_PATH ) then
    setenv LD_LIBRARY_PATH "${pysite}/pyferret"
else
    if ( "${LD_LIBRARY_PATH}" !~ "${pysite}/pyferret*" ) then
        setenv LD_LIBRARY_PATH "${pysite}/pyferret:${LD_LIBRARY_PATH}"
    endif
endif

## Faddpath: a tool to quickly add paths to the search lists
alias Faddpath 'if ( "\!*" != "" ) then \
                   setenv FER_GO "$FER_GO \!*" \
                   setenv FER_DATA "$FER_DATA \!*" \
                   setenv FER_DESCR "$FER_DESCR \!*" \
                   setenv FER_GRIDS "$FER_GRIDS \!*" \
                else \
                   echo "    Usage: Faddpath new_directory_1 ... " \
                endif'

