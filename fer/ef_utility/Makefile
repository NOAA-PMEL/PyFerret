#
# Makefile (for the FERRET/fer/efn_ext code)
#
# August 11 1997
# Jonathan Callahan (with some help from Joe Sirott)
#

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ./platform_specific_flags.mk.$(HOSTTYPE)
include ./SOURCE_FILES
include ./LIB_NAME


#
# Macros
#

OBJS = $(SRCS_F:.F=.o)
C_OBJS = EF_Util.o


#
# Targets
#

all:	ferret_cmn tmap_format $(LIB)


$(LIB): $(OBJS) $(C_OBJS)

ferret_cmn:
	ln -s ../common ./ferret_cmn

tmap_format:
	ln -s ../../fmt/cmn ./tmap_format

clean:
	rm -f core *.o *.a ferret_cmn tmap_format

#
# End of Makefile
#
