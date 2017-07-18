#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../../site_specific.mk
include ../../platform_specific.mk.$(BUILDTYPE)
include ./SOURCE_FILES

#
# Macros
#

OBJS = $(SRCS_C:.c=.o) $(SRCS_F:.F=.o)

# change the standard rules for .F -> .o builds so $(PPLUS_FFLAGS) is used
# Directly compile the .F source files to the .o object files
# since gfortran can handle the C compiler directives in Fortran code
%.o : %.F
	$(FC) $(PPLUS_FFLAGS) -c $*.F -o $*.o


#
# Targets
#

all:	$(OBJS)

clean:
	rm -f *.o

#
# End of Makefile
#
