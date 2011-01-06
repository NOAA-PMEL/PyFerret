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

.F.so:
	$(F77) $(FFLAGS) $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so 

.c.o:
	cc ${CFLAGS} $<

.f.o:
	$(F77) $(FFLAGS) $<

#
# Targets
#

all:	efstrings.so maxstrlen.so pass_thru_string.so pick_a_string.so string_cat3.so string_len_sum.so strings_as_args.so length_of_strings.so


debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
