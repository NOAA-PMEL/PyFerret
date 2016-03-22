#
# Site-specific defines
#
include ../../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../../platform_specific.mk.$(BUILDTYPE)

include ./SOURCE_FILES
include ./LIB_NAME

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

all:	$(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	$(RANLIB) $@

clean:
	rm -f *.o $(LIB)

#
# End of Makefile
#
