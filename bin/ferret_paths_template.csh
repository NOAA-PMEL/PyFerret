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

## Web browser for your system used in some "go" scripts
setenv FER_WEB_BROWSER "firefox"

## =========== The remainder of this file should not need modification ===========
## =========== unless you want to add custom directories or sites to   ===========
## =========== the Ferret's defaults.                                  ===========


## Prepend ${FER_DIR}/bin to ${PATH}
## System Manager: If you prefer not to modify PATH here, you may comment
## out these lines and execute the file $FER_DIR/bin/install_ferret_links
## which will create ferret links in /usr/local/bin.
if ( "${PATH}" !~ "*${FER_DIR}/bin*" ) then
    setenv PATH "${FER_DIR}/bin:${PATH}"
    rehash
endif

## Space-separated lists of directories examined when searching
## for (data, descriptor, grid, go-script) files without path components
setenv FER_DATA ". ${FER_DSETS}/data ${FER_DIR}/go ${FER_DIR}/examples"
setenv FER_DESCR ". ${FER_DSETS}/descr"
setenv FER_GRIDS ". ${FER_DSETS}/grids"
setenv FER_GO ". ${FER_DIR}/go ${FER_DIR}/examples ${FER_DIR}/contrib"

## Space-separated list of directories containing traditional
## Ferret external function files (shared-object libraries)
setenv FER_EXTERNAL_FUNCTIONS "${FER_DIR}/ext_func/libs"

## Space-separated list of directories for Ferret color palettes
setenv FER_PALETTE ". ${FER_DIR}/ppl"
## Ferret's color palettes directory (old)
setenv SPECTRA "${FER_DIR}/ppl"

## Directory for Ferret fonts
setenv FER_FONTS "${FER_DIR}/ppl/fonts"
## Directory for Ferret fonts (old)
setenv PLOTFONTS "${FER_DIR}/ppl/fonts"

## Faddpath: a tool to quickly add paths to the search lists
alias Faddpath 'if ( "\!*" != "" ) then \
                   setenv FER_GO "$FER_GO \!*" \
                   setenv FER_DATA "$FER_DATA \!*" \
                   setenv FER_DESCR "$FER_DESCR \!*" \
                   setenv FER_GRIDS "$FER_GRIDS \!*" \
                else \
                   echo "    Usage: Faddpath new_directory_1 ... " \
                endif'

