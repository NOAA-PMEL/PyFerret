#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
 
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so

EF_OBJS = ef_utility/*.o

#
# Rules
#

.F.so:
	$(F77) $(FFLAGS) -c $<
	$(LD) $(LD_DYN_FLAGS) $(EF_OBJS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#

all:    ef_utility ferret_cmn compressi.so compressj.so compressk.so compressl.so  samplei.so samplej.so samplek.so samplel.so samplexy.so samplet_date.so sorti.so sortj.so sortk.so sortl.so

ef_utility:
	ln -s ../ef_utility ef_utility

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_EXTERNAL_FUNCTIONS)

clean:
	-rm -f *.o *.so core a.out temp.* ef_utility ferret_cmn

#
# End of Makefile
#
