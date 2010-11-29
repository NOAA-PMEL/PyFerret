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
	$(F77) $(FFLAGS) -c  $<
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so
 

#
# Targets
#

all:	ferret_cmn add_9.so avet.so pass_thru.so percent_good_t.so storage.so string_arg.so subtract.so factorial.so dates.so

 
new:	ferret_cmn pass_thru_string.so pick_a_string.so strings_as_args.so stringlengths.so
 
extra:	ferret_cmn custom.so extend.so  status.so  strings.so times_table.so

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug" "CFLAGS = $(CFLAGS) -g -Ddebug"  all


ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
