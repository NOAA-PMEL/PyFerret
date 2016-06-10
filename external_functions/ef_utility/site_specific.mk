## Site-dependent definitions included in Makefiles

## Machine type for which to build the local external functions
## Use $(HOSTTYPE) to build natively for the machine you are using
# BUILDTYPE	= $(HOSTTYPE)
# BUILDTYPE	= x86_64-linux
# BUILDTYPE	= i386-linux
BUILDTYPE	= intel-mac

## Installation directory for the locally built external functions
FER_LOCAL_EXTFCNS = $(FER_DIR)/ext_func/libs

