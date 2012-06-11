#! /bin/csh
## This C-shell script is used to initialize and run Ferret using the
## pyferret Python module.  The intent of this script is to provide a
## traditional Ferret interface through the pyferret module.


## Assign the python2.x executable to use
set pyname = "python2.6"


## Make sure the FER_* environment variables are assigned
if ( ! $?FER_LIBS ) then
## Either source the ferret_paths script to assign the environment variables
#    source "/my/path/to/ferret_paths.csh"
## or just throw an error if they should have already been defined
    echo "**ERROR: Ferret environment variables are not defined"
    exit 1
##
endif


## Assign the directory containing the pyferret Python package (directory)
## One of:
## (a) the default user-specific site
##     Defining pysite not required if Python 2.6 or later.
# set pysite = "${HOME}/.local/lib/${pyname}/site-packages"
## or:
## (b) the default system-wide site, typically one of:
##         /usr/lib/python2.x/site-packages
##         /usr/local/lib/python2.x/site-packages
##         /usr/lib64/python2.x/site-packages
##         /usr/local/lib64/python2.x/site-packages
##     Run: 
##        python2.x -c "import sys; print sys.path"
##     to show locations searched.
# set pysite = "/usr/local/lib/${pyname}/site-packages"
## or:
## (c) a custom directory (for example, if installed under $FER_DIR)
##     Defining pysite required.
set pysite = "${FER_LIBS}/${pyname}/site-packages"
##
## set pysite = "${FER_LIBS}/${pyname}/site-packages"
## should be correct for the normal Ferret installation.


##
## ==== The following should not need any modifications ====
##


## Add $pysite to the Python search path given by PYTHONPATH
if ( ! $?PYTHONPATH ) then
    setenv PYTHONPATH "${pysite}"
else
    if ( "${PYTHONPATH}" !~ "*${pysite}*" ) then
        setenv PYTHONPATH "${pysite}:${PYTHONPATH}"
    endif
endif


## Add $pysite/pyferret to the shared-object library search path given by LD_LIBRARY_PATH
if ( ! $?LD_LIBRARY_PATH ) then
    setenv LD_LIBRARY_PATH "${pysite}/pyferret"
else
    if ( "${LD_LIBRARY_PATH}" !~ "*${pysite}/pyferret*" ) then
        setenv LD_LIBRARY_PATH "${pysite}/pyferret:${LD_LIBRARY_PATH}"
    endif
endif


## Finally, execute an in-line Python script to run Ferret using the pyferret 
## module.  The following explicity processes the $PYTHONSTARTUP file, if it
## exists and if '-secure' was not given as a command-line argument.
${pyname} -i -c "\
import sys; \
import os; \
import rlcompleter; \
import readline; \
import pyferret; \
readline.parse_and_bind('tab: complete'); \
if not '-secure' in sys.argv[1:]: \
    try: \
        execfile(os.getenv('PYTHONSTARTUP', '')); \
    except IOError: \
        pass; \
(errval, errmsg) = pyferret.init(sys.argv[1:], True)" $*

