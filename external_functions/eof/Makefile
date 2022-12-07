#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
#  ACM  2/2001  debug macros
#  ACM  Jul 2001 Move to statically linked code in fer/efi.  name functions
#                eofspace, eofstat, eoftfunc; with same code as in fer/efi,
#                but with different INCLUDE statements for use in testing
#                as dynamically-linked functions.
# 12/2022 Remove from targets list routines and functions that
#         are linked into the PyFerret program

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

#   once eof_space etc are linked into Ferret dont need SUB_OBJS

.F.so:
	$(FC) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

all:	eofsubs.o eof_space_tgap.so eof_tfunc_tgap.so eof_stat_tgap.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

# To make a tarfile with the older functions that work on gappy timeseries:
# make; make tarfile
tarfile:
	rm -f fer_eof_gappy_timeseries_fcns.tar*
	tar cf fer_eof_gappy_timeseries_fcns.tar *.so
	gzip -9 fer_eof_gappy_timeseries_fcns.tar

#
# End of Makefile
#
