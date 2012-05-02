#
# Makefile for the ppl libraries
#

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../site_specific.mk
include ../platform_specific.mk.$(BUILDTYPE)

DIRS	= tmapadds complot ourlib plot \
	  plotlib pplepic pplusr symlib

# change the standard rules for .F -> .o builds so $(PPLUS_FFLAGS) is used
%.o : %.F
	rm -f $*.f
	$(CPP) -P -traditional $(CPP_FLAGS) $(<F) | sed -e 's/de    /de /g' | sed -e 's/de         /de /g' > $*.f
	$(F77) $(PPLUS_FFLAGS) -c $*.f	 

#
# Targets
#

all:
	for i in $(DIRS) ; do \
            if ! $(MAKE) -C $$i ; then \
                exit 1 ; \
            fi ; \
        done


clean:
	for i in $(DIRS) ; do \
            $(MAKE) -C $$i clean ; \
        done


#
# End of Makefile
#
