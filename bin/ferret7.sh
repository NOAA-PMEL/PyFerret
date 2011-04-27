#! /bin/sh
## This Bourne shell script is used to set environment variables need by
## pyferret, and then execute ferret.py in python.  The intent of this
## script is to provide a traditional Ferret interface through pyferret.


## Assign the python2.x executable to use
pyname="python2.6"


## Make sure the FER_* environment variables are assigned
if [ -z "$FER_DIR" ]; then
##   Either source the ferret_paths script to assign the environment variables
#    . "/my/path/to/ferret_paths.sh"
##   or just throw an error if they should have already been defined
    echo "**ERROR: Ferret environment variables are not defined"
    exit 1
##
fi


## Assign the Python directory containing the pyferret package and ferret.py
## One of:
## (a) the default user-specific site
##     Defining pysite not required if Python 2.6 or later.
# pysite="${HOME}/.local/lib/${pyname}/site-packages"
## or:
## (b) the default system-wide site, typically one of:
##         /usr/lib/python2.x/site-packages
##         /usr/local/lib/python2.x/site-packages
##         /usr/lib64/python2.x/site-packages
##         /usr/local/lib64/python2.x/site-packages
##     Run: 
##        python2.x -c "import sys; print sys.path"
##     to show locations searched.
##     Do not define pysite.
## or:
## (c) a custom directory (for example, if installed under $FER_DIR)
##     Defining pysite required.
pysite="${FER_DIR}/lib/${pyname}/site-packages"


## Add pysite to the Python search path given by PYTHONPATH
if [ -n "$pysite" ]; then
    if [ -z "$PYTHONPATH" ]; then
        export PYTHONPATH="${pysite}"
    else
        if ! echo "${PYTHONPATH}" | grep -q "${pysite}"; then
            export PYTHONPATH="${pysite}:${PYTHONPATH}"
        fi
    fi
fi


## Finally, start pyferret via the ferret.py script
${pyname} -i -c "import sys; import pyferret; (errval, errmsg) = pyferret.init(sys.argv[1:], True)" $*

