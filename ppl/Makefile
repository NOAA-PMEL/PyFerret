#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../site_specific.mk
include ../platform_specific.mk.$(BUILDTYPE)

DIRS	= tmapadds complot ourlib plot \
	  plotlib pplepic pplusr symlib

# change the standard rules for .F -> .o builds so $(PPLUS_FFLAGS) is used
# Directly compile the .F source files to the .o object files
# since gfortran can handle the C compiler directives in Fortran code
%.o : %.F
	$(FC) $(PPLUS_FFLAGS) -c $*.F -o $*.o

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
