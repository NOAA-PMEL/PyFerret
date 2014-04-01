#
# Makefile for Ferret External Functions
#
# *sh* 12/27/00
# *sh* 1/02 -- updated to add expndi_by
# *sh* 2/02 -- updated to add expndi_by_z
# *acm*2/02 -- cmprsi_by has been replaced by family of compress_by
#              functions, which are internally linked.
#

#
# include site specific macro definitions
#
include ../ef_utility/site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

# all:	expndi_by.so expndi_by_z.so expndi_by_t.so cmprsi_by.so expndi_by_z_counts.so 
all:	

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
