#
# Site-specific defines
#
include ../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../platform_specific.mk.$(BUILDTYPE)

#
# The shared-object library ferret_ef_mem_subsc.so contains an external
# copy of the common block FERRET_EF_MEM_SUBSC in a function which copies
# the values from Ferret's internal copy of the common block.  These values
# are obtained by calling the Ferret functions ef_get_arg_mem_subscripts,
# ef_get_res_mem_subscripts, and ef_get_wrk_mem_subscripts.
#
# This shared-object library is loaded using dlopen with the RTLD_GLOBAL
# flag so this external copy of the common block can be seen by all Ferret
# Fortran external functions.  Python import uses the RTLD_LOCAL flag with
# dlopen (for encapsulation), thus the reason Ferret's internal copy of the
# common block is not seen by the Fortran external functions.
#
# The name intentionally does not start with "lib" since this library should
# not be linked with anything.  Ferret loads it dynamically with dlopen.
EFMEMLIB = ../lib/ferret_ef_mem_subsc.so

#
# Targets
#

all: $(EFMEMLIB)

$(EFMEMLIB): copy_ferret_ef_mem_subsc.o
	$(LD) $(LD_DYN_FLAGS) copy_ferret_ef_mem_subsc.o \
	-L ../pyferret_install/lib/$(PYTHON_EXE)/site-packages/pyferret \
	-L ../pyferret_install/lib64/$(PYTHON_EXE)/site-packages/pyferret \
	-lpyferret -o $(EFMEMLIB)

clean:
	rm -f *.o $(EFMEMLIB)

#
# End of Makefile
#
