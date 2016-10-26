##
## Environment settings for Ferret for csh users.
## Source this file before running ferret ('source ferret_paths.csh')
##

## The environment variable FER_DIR should be the pathname of
## the directory you created for the FERRET software.
setenv FER_DIR "/usr/local/ferret"

## The environment variable FER_DSETS should be the pathname of
## the directory you created for the FERRET demonstration data files (30+ Mbytes).
setenv FER_DSETS "${FER_DIR}/fer_dsets"

## set python_exe to the (optionally full-path) python executable to use
set python_exe = PYTHON_EXECUTABLE

## set python_subdir to 'python2.6' or 'python2.7' 
## whichever is appropriate for the above python executable
set python_subdir = PYTHON_SUBDIRECTORY

## Web browser for your system used in some "go" scripts
setenv FER_WEB_BROWSER "firefox"

## If "java -version" does not run from the command prompt, or
## does not report a java 1.6.x or later (e.g., 1.7.x) version, 
## the environment variable JAVA_HOME needs to be defined in 
## order to run the ThreddsBrowser GUI.  The directory defined 
## by this environment variable contains the java executable 
## in the bin subdirectory (ie, bin/java).
if ( ! $?JAVA_HOME ) then
    ## try some common locations
    if ( -x "/usr/java/latest/bin/java" ) then
        setenv JAVA_HOME "/usr/java/latest"
    else if ( -x "/usr/lib/jvm/java-1.7.0/bin/java" ) then
        setenv JAVA_HOME "/usr/lib/jvm/java-1.7.0"
    else if ( -x "/usr/lib/jvm/java-7/bin/java" ) then
        setenv JAVA_HOME "/usr/lib/jvm/java-7"
    else if ( -x "/usr/lib/jvm/java-1.6.0/bin/java" ) then
        setenv JAVA_HOME "/usr/lib/jvm/java-1.6.0"
    else if ( -x "/usr/lib/jvm/java-6/bin/java" ) then
        setenv JAVA_HOME "/usr/lib/jvm/java-6"
    endif
    ## or comment the above out and just set your own location
    # setenv JAVA_HOME "/my/java/home"
endif

## =========== The remainder of this file should not need modification ===========
## =========== unless you want to add custom directories or sites to   ===========
## =========== the Ferret's defaults.                                  ===========


## Prepend ${FER_DIR}/bin to ${PATH}
## System Manager: If you prefer not to modify PATH here, you may comment
## out these lines and execute the file $FER_DIR/bin/install_ferret_links
## which will create ferret links in /usr/local/bin.
if ( "${PATH}" !~ "${FER_DIR}/bin:*" ) then
    setenv PATH "${FER_DIR}/bin:${PATH}"
    rehash
endif

## Space-separated list of default sites for ThreddsBrowser
## (SET /DATA /BROWSE or its alias OPEN)
## Assigned in this unusual way to make it easy to add/delete/rearrange sites.
setenv FER_DATA_THREDDS ""
setenv FER_DATA_THREDDS "${FER_DATA_THREDDS} http://ferret.pmel.noaa.gov/geoide/geoIDECleanCatalog.xml"
setenv FER_DATA_THREDDS "${FER_DATA_THREDDS} ${FER_DSETS}"

## Space-separated lists of directories examined when searching
## for (data, descriptor, grid, go-script) files without path components
setenv FER_DATA ". ${FER_DSETS}/data ${FER_DIR}/go ${FER_DIR}/examples"
setenv FER_DESCR ". ${FER_DSETS}/descr"
setenv FER_GRIDS ". ${FER_DSETS}/grids"
setenv FER_GO ". ${FER_DIR}/go ${FER_DIR}/examples ${FER_DIR}/contrib"

## Space-separated list of directories containing traditional
## Ferret external function files (shared-object libraries)
# setenv FER_EXTERNAL_FUNCTIONS "${FER_DIR}/ext_func/libs"
## PyFerret external function files (shared-object libraries)
setenv PYFER_EXTERNAL_FUNCTIONS "${FER_DIR}/ext_func/pylibs"

## Space-separated list of directories for Ferret color palettes
setenv FER_PALETTE ". ${FER_DIR}/ppl"
## Ferret's color palettes directory (old)
setenv SPECTRA "${FER_DIR}/ppl"

## Directory for Ferret fonts
setenv FER_FONTS "${FER_DIR}/ppl/fonts"
## Directory for Ferret fonts (old)
setenv PLOTFONTS "${FER_DIR}/ppl/fonts"

## Directory containing threddsBrowser.jar and toolsUI.jar for ThreddsBrowser
setenv FER_LIBS "${FER_DIR}/lib"

setenv FER_DAT "${FER_DIR}"

## Assign the directory containing the pyferret Python package (directory)
set pysite = "${FER_LIBS}/${python_subdir}/site-packages"

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
## by LD_LIBRARY_PATH so libpyferret.so will be found.
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

