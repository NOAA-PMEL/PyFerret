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
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#

# all:	ferret_cmn expndi_by.so expndi_by_z.so expndi_by_t.so
all:	ferret_cmn 

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
