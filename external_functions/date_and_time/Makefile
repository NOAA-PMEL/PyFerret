#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
# dec-00 adapted for "godae_obs" function


#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
include date_and_time_platform_specific_flags.mk.$(HOSTTYPE)

#
# Macros
#

.SUFFIXES: .so

DAT_OBJS = date_decode.o time_decode.o

#
# Rules
#

.c.o:
	cc ${DATFLAGS} ${CFLAGS} -c $<

.F.o:
	$(F77) $(FFLAGS) -u -c $<

.o.so:
	$(LD) $(LD_DYN_FLAGS) $(DAT_OBJS) $*.o -o $*.so 

#
# Targets
#

all:	ferret_cmn date_decode.o time_decode.o date1900.o date1900.so minutes24.o minutes24.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -C -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
