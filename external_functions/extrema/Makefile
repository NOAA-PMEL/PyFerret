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
include ../../site_specific.mk

#
# include platform specific macro definitions
#
include ../../platform_specific.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.o: 
	$(F77) $(FFLAGS) -g -c $<

.F.so:
	$(F77) $(FFLAGS)  $<
	$(LD) $(LD_DYN_FLAGS) $*.o minminmax.o maxminmax.o -o $*.so

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
