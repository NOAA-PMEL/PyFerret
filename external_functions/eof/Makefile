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
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

SUB_OBJS = eofsubs.o 

#
# Rules
#

#   once eof_space etc are linked into Ferret dont need SUB_OBJS
.f.so:
	$(F77) $(FFLAGS)  $<
	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o -o $*.so 


#.f.so:
#	$(F77) $(FFLAGS)  $<
#	$(LD) $(LD_DYN_FLAGS) $(SUB_OBJS) $*.o -o $*.so 

#
# Targets
#

# all:	ferret_cmn eofsubs.o eoft.so eof_space.so eof_tfunc.so eof_stat.so
all:	ferret_cmn 

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
