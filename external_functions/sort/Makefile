#
# Makefile for Ferret External Functions
#
# January 20 1998
# Jonathan Callahan
#
#  15-Nov-1999 Ansley Manke  
#  remove refs to ef_utility/*.o  (now in ferret executable)
# *sh* 2/2000 - removed SAMPLEIJ from target list (replaced by SAMPLEXY)
#
# *acm* 1/2001 - incorporate "make debug" as per SH Makefiles
#
# include platform specific macro definitions
#

include ../ef_utility/platform_specific_flags.mk.$(HOSTTYPE)
 

#
# Macros
#

.SUFFIXES: .so .o

#
# Rules
#

.F.o:
	$(F77) $(FFLAGS) $<

.o.so:
	$(LD) $(LD_DYN_FLAGS) $(SYSLIBS) $*.o -o $*.so

#
# Targets
#
all:    ferret_cmn compressi.so compressj.so compressk.so compressl.so 

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) -g -Ddebug"   "CFLAGS = $(CFLAGS) -g -Ddebug"  all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so core a.out temp.* ferret_cmn

#
# End of Makefile
#
