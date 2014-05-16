#
# Makefile for Ferret External Functions
#
#
# include platform specific macro definitions
#
include ../ef_utility/site_specific.mk
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)
 
#
# Macros
#

.SUFFIXES: .so
SUB_OBJS = piecewise_subs.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $(SUB_OBJS) $*.o -o $*.so

#
# Targets
#

#all:	$(SUB_OBJS) piecewise.so piecewise3.so piecewise1.so piecewise2.so separate.so sample_fast_i.so sample_rows.so apply_qc.so
all:	$(SUB_OBJS) piecewise3.so separate.so sample_fast_i.so sample_fast_i_str.so sample_rows.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
