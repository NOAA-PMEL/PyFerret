#
# Makefile (for the FERRET/fer/ef_utility code)
#
# August 11 1997
# Jonathan Callahan (with some help from Joe Sirott)
#
# Ansley Manke 11/99 NOTE:
#    We DO NOT want to make the ef_utility/*.o files into a libaray.
#    The ferret Makefile is to link these object files in the executable.
#    Because they are only called by external functions they are not seen 
#    as unresolved references and therefore not searched for if they are in 
#    a library.

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../platform_specific_flags.mk.$(HOSTTYPE)
include ./SOURCE_FILES
include ./SOURCE_FILES_C

#
# Macros
#

OBJS = $(SRCS_C:.c=.o) $(SRCS_F:.F=.o)

#
# Targets
#

all:	ferret_cmn tmap_format $(OBJS)

ferret_cmn:
	ln -s ../common ./ferret_cmn

tmap_format:
	ln -s ../../fmt/cmn ./tmap_format

clean:
	rm -f core *.o *.a ferret_cmn tmap_format

#
# End of Makefile
#
