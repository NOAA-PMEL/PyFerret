#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so
 

#
# Targets
#

standard:	ferret_cmn add_9.so avet.so pass_thru.so percent_good_t.so subtract.so

all:	ferret_cmn add_9.so avet.so custom.so custom_nf.so extend.so pass_thru.so percent_good_t.so status.so storage.so strings.so subtract.so times_table.so

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
