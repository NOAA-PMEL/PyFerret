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

EF_LIBS      = ./ef_util/libef_util.a

#
# Rules
#

.F.o:
	$(F77) $(FFLAGS) -c $<

#
# Targets
#

all:	utility avet.so add_9.so subtract.so pass_thru.so times_table.so

add_9.so: add_9.o $(EF_LIBS)
	$(LD) $(LD_DYN_FLAGS) $? -o $@ $(EF_LIBS)
	cp $@ ./myfuncs
 
avet.so: avet.o $(EF_LIBS)
	$(LD) $(LD_DYN_FLAGS) $? -o $@ $(EF_LIBS)
	cp $@ ./myfuncs
 
times_table.so: times_table.o $(EF_LIBS)
	$(LD) $(LD_DYN_FLAGS) $? -o $@ $(EF_LIBS)
	cp $@ ./myfuncs
 
pass_thru.so: pass_thru.o $(EF_LIBS)
	$(LD) $(LD_DYN_FLAGS) $? -o $@ $(EF_LIBS)
	cp $@ ./myfuncs
 
subtract.so: subtract.o $(EF_LIBS)
	$(LD) $(LD_DYN_FLAGS) $? -o $@ $(EF_LIBS)
	cp $@ ./myfuncs


utility:
	(cd ef_util; $(MAKE) all);
 
clean:
	(cd ef_util; $(MAKE) clean);
	-rm -f *.o *.so core a.out temp.* ./myfuncs/* 

#
# End of Makefile
#
