#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
# *sh* 2/2000 - removed SAMPLEIJ from target list (replaced by SAMPLEXY)
#
# *acm* 1/2001 - incorporate "make debug" as per SH Makefiles
#  acm  2/2001   debug macros
#  acm  5/2007   removed fill_xy from target list; now statically-linked.

#
# include site-specific definitions
#
include ../../site_specific.mk

#
# include platform-specific definitions
#
include ../../platform_specific.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so

#
# Targets
#

#all:    chdir.so neutral.so fill_xy.so pco2.so
all:    chdir.so neutral.so fill_xy.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
