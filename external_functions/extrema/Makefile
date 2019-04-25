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
# ACM 2/2001  debug macros

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

.F.o:
	$(FC) $(FFLAGS) -c $<

.F.so:
	$(FC) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $*.o minminmax.o maxminmax.o $(SYSLIBS) -o $*.so

#
# Targets
#

# all:    minminmax.o maxminmax.o findhi.so findlo.so
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
