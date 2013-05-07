#! /bin/csh
## This C-shell script is used to initialize and run Ferret using the 
## pyferret Python module.  The intent of this script is to provide a 
## traditional Ferret interface through the pyferret module.

## set python_exe to the (optionally full-path) python executable to use
set python_exe = PYTHON_EXECUTABLE

## set python_subdir to 'python2.6' or 'python2.7' 
## whichever is appropriate for the above python executable
set python_subdir = PYTHON_SUBDIRECTORY

## Make sure the FER_* environment variables are assigned
if ( ! $?FER_LIBS ) then
## Either source the ferret_paths script to assign the environment variables
#    source "/my/path/to/ferret_paths.csh"
## or just throw an error if they should have already been defined
    echo "**ERROR: Ferret environment variables are not defined"
    exit 1
##
endif

##
## ==== The following should not need any modifications ====
##


## Assign the directory containing the pyferret Python package (directory)
set pysite = "${FER_LIBS}/${python_subdir}/site-packages"

## Add $pysite to the Python search path given by PYTHONPATH 
## so the pyferret package will be found.
if ( ! $?PYTHONPATH ) then
    setenv PYTHONPATH "${pysite}"
else
    if ( "${PYTHONPATH}" !~ "*${pysite}*" ) then
        setenv PYTHONPATH "${pysite}:${PYTHONPATH}"
    endif
endif

## Add $pysite/pyferret to the shared-object library search path given 
## by LD_LIBRARY_PATH so libpyferret.so will be found.
if ( ! $?LD_LIBRARY_PATH ) then
    setenv LD_LIBRARY_PATH "${pysite}/pyferret"
else
    if ( "${LD_LIBRARY_PATH}" !~ "*${pysite}/pyferret*" ) then
        setenv LD_LIBRARY_PATH "${pysite}/pyferret:${LD_LIBRARY_PATH}"
    endif
endif

## Finally, execute an in-line Python script to run Ferret using the pyferret 
## module.  The init method explicity processes the $PYTHONSTARTUP file, if it
## exists and if '-secure' was not given as a command-line argument.
${python_exe} -i -c "import sys; import pyferret; (errval, errmsg) = pyferret.init(sys.argv[1:], True)" $*

