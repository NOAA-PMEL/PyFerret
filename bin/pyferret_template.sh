#! /bin/sh
## This Bourne-shell script is used to initialize and run Ferret using 
## the pyferret Python module.  The intent of this script is to provide 
## a traditional Ferret interface through the pyferret module.

## set python_exe to the (optionally full-path) python executable to use
python_exe=PYTHON_EXECUTABLE

## set python_subdir to 'python2.6' or 'python2.7' 
## whichever is appropriate for the above python executable
python_subdir=PYTHON_SUBDIRECTORY

## Make sure the FER_* environment variables are assigned
if [ -z "${FER_LIBS}" ]; then
## Either source the ferret_paths script to assign the environment variables
#    . "/my/path/to/ferret_paths.sh"
## or just throw an error if they should have already been defined
    echo "**ERROR: Ferret environment variables are not defined"
    exit 1
##
fi


##
## ==== The following should not need any modifications ====
##


## Assign the directory containing the pyferret Python package (directory)
pysite="${FER_LIBS}/${python_subdir}/site-packages"

## Add pysite to the Python search path given by PYTHONPATH
## so the pyferret package will be found.
if [ -z "${PYTHONPATH}" ]; then
    export PYTHONPATH="${pysite}"
else
    if ! echo "${PYTHONPATH}" | grep -q "${pysite}"; then
        export PYTHONPATH="${pysite}:${PYTHONPATH}"
    fi
fi

## Add $pysite/pyferret to the shared-object library search path given 
## by LD_LIBRARY_PATH so libpyferret.so will be found.
if [ -z "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${pysite}/pyferret"
else
    if ! echo "${LD_LIBRARY_PATH}" | grep -q "${pysite}/pyferret"; then
        export LD_LIBRARY_PATH="${pysite}/pyferret:${LD_LIBRARY_PATH}"
    fi
fi

## Finally, execute an in-line Python script to run Ferret using the pyferret 
## module.  The init method explicity processes the $PYTHONSTARTUP file, if it
## exists and if '-secure' was not given as a command-line argument.
${python_exe} -i -c "import sys; import pyferret; (errval, errmsg) = pyferret.init(sys.argv[1:], True)" $*

