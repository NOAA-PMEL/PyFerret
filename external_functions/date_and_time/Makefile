#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
# dec-00 adapted for "date_and_time" function
# 2/01 *sh* - restored the -C qualifier for debug target

#
# include site specific macro definitions
#
include ../../site_specific.mk

#
# include platform specific macro definitions
#
include ../ef_utility/platform_specific.mk.$(BUILDTYPE)

#
# Macros
#

.SUFFIXES: .so

DAT_OBJS = date_decode.o time_decode.o secs_to_ymdhms.o

#
# Rules
#

.c.so:
	$(CC) ${CFLAGS} -c $<
	$(LD) $(LD_DYN_FLAGS) $(DAT_OBJS) $*.o -o $*.so

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(DAT_OBJS) $*.o -o $*.so

#
# Targets
#

# all:	date_decode.o time_decode.o secs_to_ymdhms.o date1900.o date1900.so minutes24.o minutes24.so days1900toymdhms.o days1900toymdhms.so

all:	

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so

#
# End of Makefile
#
