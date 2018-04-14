##
## Environment settings for Ferret for Bourne-shell users.
## Source this file before running ferret ('. ferret_paths.sh')
##

## The environment variable FER_DIR should be the pathname of
## the directory you created for the FERRET software.
export FER_DIR="/usr/local/ferret"

## The environment variable FER_DSETS should be the pathname of
## the directory you created for the FERRET demonstration data files (30+ Mbytes).
export FER_DSETS="${FER_DIR}/fer_dsets"


## =========== The remainder of this file should not need modification ===========
## =========== unless you want to add custom directories or sites to   ===========
## =========== the Ferret's defaults.                                  ===========


## Add ${FER_DIR}/bin to the beginning of ${PATH}
if ! echo "${PATH}" | grep -q "${FER_DIR}/bin"; then
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

## Space-separated lists of directories examined when searching
## for data, descriptor, grid, go-script files without path components
export FER_DATA=". ${FER_DSETS}/data ${FER_DIR}/contrib"
export FER_DESCR=". ${FER_DSETS}/descr"
export FER_GRIDS=". ${FER_DSETS}/grids"
export FER_GO=". ${FER_DIR}/go ${FER_DIR}/examples ${FER_DIR}/contrib"

## Space-separated list of directories containing traditional
## Ferret external function files (shared-object libraries)
export FER_EXTERNAL_FUNCTIONS="${FER_DIR}/ext_func/libs"

## Space-separated list of directories for Ferret color palettes
export FER_PALETTE=". ${FER_DIR}/ppl"
## Ferret's color palettes directory (old)
export SPECTRA="${FER_DIR}/ppl"

## Directory for Ferret fonts
export FER_FONTS="${FER_DIR}/ppl/fonts"
## Directory for Ferret fonts (old)
export PLOTFONTS="${FER_DIR}/ppl/fonts"

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

