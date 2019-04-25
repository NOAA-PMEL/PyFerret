#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
#  ACM 2/2001  debug macros
#  ACM 2/2002  change targets to all and extras; somehow standard
#               not working well w/ linux.

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
	$(FC) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

all:	add_9.so appende.so avet.so dates.so factorial.so pass_thru.so percent_good_t.so \
	storage.so string_arg.so subtract.so earth_distance.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
