##
## Environment settings for Ferret for bash users.
## Source this file before running ferret ('. ferret_paths.sh')
##

## The environment variable FER_DIR should be the pathname of
## the directory you created for the FERRET software.
export FER_DIR="/usr/local/ferret"

## The environment variable FER_DSETS should be the pathname of
## the directory you created for the FERRET demonstration data files (30+ Mbytes).
export FER_DSETS="${FER_DIR}/fer_dsets"

## set python_exe to the (optionally full-path) python executable to use
python_exe=PYTHON_EXECUTABLE

## set python_subdir to 'python2.6' or 'python2.7' 
## whichever is appropriate for the above python executable
python_subdir=PYTHON_SUBDIRECTORY

## Web browser for your system used in some "go" scripts
export FER_WEB_BROWSER="firefox"

## If "java -version" does not run from the command prompt, or
## does not report a java 1.6.x or later (e.g., 1.7.x) version, 
## the environment variable JAVA_HOME needs to be defined in 
## order to run the ThreddsBrowser GUI.  The directory defined 
## by this environment variable contains the java executable 
## in the bin subdirectory (ie, bin/java).
if [ -z "$JAVA_HOME" ]; then
    ## try some common locations; 
    if [ -x "/usr/java/latest/bin/java" ]; then
        export JAVA_HOME="/usr/java/latest"
    elif [ -x "/usr/lib/jvm/java-1.7.0/bin/java" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-1.7.0"
    elif [ -x "/usr/lib/jvm/java-7/bin/java" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-7"
    elif [ -x "/usr/lib/jvm/java-1.6.0/bin/java" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-1.6.0"
    elif [ -x "/usr/lib/jvm/java-6/bin/java" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-6"
    fi
    ## or comment the above out and just set your own location
    # export JAVA_HOME="/my/java/home"
fi


## =========== The remainder of this file should not need modification ===========
## =========== unless you want to add custom directories or sites to   ===========
## =========== the Ferret's defaults.                                  ===========


## Prepend ${FER_DIR}/bin to ${PATH}
## System Manager: If you prefer not to modify PATH here, you may comment
## out these lines and execute the file $FER_DIR/bin/install_ferret_links
## which will create ferret links in /usr/local/bin.
if ! echo "${PATH}" | grep -q "^${FER_DIR}/bin:"; then
    export PATH="${FER_DIR}/bin:${PATH}"
fi

## Space-separated list of default sites for ThreddsBrowser
## (SET /DATA /BROWSE or its alias OPEN)
## Assigned in this unusual way to make it easy to add/delete/rearrange sites.
export FER_DATA_THREDDS=""
export FER_DATA_THREDDS="${FER_DATA_THREDDS} http://ferret.pmel.noaa.gov/geoide/geoIDECleanCatalog.xml"
export FER_DATA_THREDDS="${FER_DATA_THREDDS} ${FER_DSETS}"

## Space-separated lists of directories examined when searching
## for (data, descriptor, grid, go-script) files without path components
export FER_DATA=". ${FER_DSETS}/data ${FER_DIR}/go ${FER_DIR}/examples"
export FER_DESCR=". ${FER_DSETS}/descr"
export FER_GRIDS=". ${FER_DSETS}/grids"
export FER_GO=". ${FER_DIR}/go ${FER_DIR}/examples ${FER_DIR}/contrib"

## Space-separated list of directories containing traditional
## Ferret external function files (shared-object libraries)
# export FER_EXTERNAL_FUNCTIONS="${FER_DIR}/ext_func/libs"
## PyFerret external function files (shared-object libraries)
export PYFER_EXTERNAL_FUNCTIONS="${FER_DIR}/ext_func/pylibs"

## Space-separated list of directories for Ferret color palettes
export FER_PALETTE=". ${FER_DIR}/ppl"
## Ferret's color palettes directory (old)
export SPECTRA="${FER_DIR}/ppl"

## Directory for Ferret fonts
export FER_FONTS="${FER_DIR}/ppl/fonts"
## Directory for Ferret fonts (old)
export PLOTFONTS="${FER_DIR}/ppl/fonts"

## Directory containing threddsBrowser.jar and toolsUI.jar for ThreddsBrowser
## as well as the python2.x/site-packages directory for PyFerret Python packages
export FER_LIBS="${FER_DIR}/lib"

## Ferret directory (old)
export FER_DAT="${FER_DIR}"

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

