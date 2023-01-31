#
# Makefile for Ferret External Functions
#
# 12/2022 Remove from targets list routines and functions that
#         are linked into the PyFerret program
#
# include platform specific macro definitions
#
include ../ef_utility/site_specific.mk
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
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o $(SYSLIBS) -o $*.so

#
# Targets
#

#all:	$(SUB_OBJS) piecewise.so piecewise3.so piecewise1.so piecewise2.so separate.so sample_fast_i.so sample_rows.so apply_qc.so sample_fast_i_str.so sample_rows.so
all:	

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
