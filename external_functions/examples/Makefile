#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
 
#
# include platform specific macro definitions
#

include platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

EFN_OBJS = efn/*.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(EFN_OBJS) $*.o -o $*.so 
	cp $@ $(FER_EXTERNAL_FUNCTIONS)


#
# Targets
#

all:	efn ferret_cmn add_9.so avet.so custom.so custom_nf.so extend.so pass_thru.so percent_good_t.so status.so storage.so storage_nxy.so strings.so string_arg.so subtract.so times_table.so

efn:
	ln -s ../efn efn

ferret_cmn:
	ln -s ../efn/ferret_cmn ferret_cmn

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
