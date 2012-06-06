#
# Site-specific defines
#
include ../../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../../platform_specific.mk.$(BUILDTYPE)

#
# The shared-object library libferret_ef_mem_subsc.so contains an external
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
EFMEMLIB = ../../lib/libferret_ef_mem_subsc.so

#
# Targets
#

all: $(EFMEMLIB)

$(EFMEMLIB): copy_ferret_ef_mem_subsc.o
	$(LD) $(LD_DYN_FLAGS) copy_ferret_ef_mem_subsc.o -o $(EFMEMLIB)

clean:
	rm -f *.o ../../lib/ferret_ef_mem_subsc.so

#
# End of Makefile
#
