#
# Makefile (for FERRET/fer/efi statically linked EF code)
#
# Feb 28, 2000  -- based on other TMAP external functions makefiles
# Ansley Manke
# Jonathan Callahan (with some help from Joe Sirott)
# 
# *acm* 3/23/2000 link  ../common to ferret_cmn  rather than using the
#       link in ef_utility (which may not be present when this is compiled.)
#
#
#  Note that these functions must be decleared in ../ccr/EF_InternalUtil.c
#  Run the perl script in this directory int_dlsym.pl ./ > intcode.c
#  put the resulting lines in three places in EF_InternalUtil.c as
#  noted in the comments at the start of that file.

#
# Site-specific defines
#
include ../../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../../platform_specific_flags.mk.$(HOSTTYPE)

include ./SOURCE_FILES
include ./LIB_NAME

#INCLUDES += -Iferret_cmn

#
# Macros
#



OBJS = $(SRCS_C:.c=.o) $(SRCS_F:.F=.o)


#
# Targets
#

all:	ferret_cmn $(LIB)

$(LIB): $(OBJS) 
	$(AR) $(ARFLAGS) $@ $(OBJS) 
	$(RANLIB) $@

ferret_cmn:
	ln -s ../common ferret_cmn

clean:
	rm -f core *.o *.a *.f ferret_cmn

#
# End of Makefile
#
