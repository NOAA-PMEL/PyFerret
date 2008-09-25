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
#  acm  2/2001   debug macros
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
	$(F77) $(FFLAGS) $<
	$(LD) $(LD_DYN_FLAGS) $*.o -o $*.so 

.c.o:
	cc ${CFLAGS} $<

.f.o:
	$(F77) $(FFLAGS) $<

#
# Targets
#
#all:    ferret_cmn bin_index_wt.so 

# all:    ferret_cmn is_element_of.so is_element_of_str.so element_index.so element_index_str.so sampleyz.so samplexz.so samplexy_closest.so samplexy_curv.so

# all:    ferret_cmn is_element_of.so samplexy_curv_avg.so samplexy_curv_nrst.so
all:    ferret_cmn samplexyt.so samplei_multi.so samplej_multi.so samplek_multi.so samplel_multi.so is_element_of_str_nocase.so 

debug:
	$(MAKE) "FFLAGS = $(FFLAGS) $(FFLAGS_DEBUG)" "CFLAGS = $(CFLAGS) $(CFLAGS_DEBUG)" all

ferret_cmn:
	ln -s ../ef_utility/ferret_cmn ferret_cmn

install:
	cp *.so $(FER_LOCAL_EXTFCNS)

clean:
	-rm -f *.o *.so *.f core a.out temp.* ferret_cmn

#
# End of Makefile
#
