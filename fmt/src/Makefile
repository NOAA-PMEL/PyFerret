# make TMAP library
# ver 0.0 3/14/91 *sh*  NOAA/PMEL/TMAP
# ver 0.1 10/11/91 kob  Jisao/pmel/tmap
# ver 1.0 2/3/93  *kob*
#        whole makefile revamped to implement recursive makes
# 10/27/94 *sh* changes for AIX: include -Dsun on the pre-processor pass
# define variables
# 8/22/95 - *kob* only have tmap_lib now, for both debug and non-debug
# 12/20/95 - # ifdef NINT_WRITE_UNMATCH added
#             to the routine TM_FMT. 
#	     For now we should **not** use NINT_WRITE_UNMATCH during
#	     compilation anywhere -- the code that needed it was
#            applicable to VAX FORTRAN.  However, if we see odd
#            behavior from TM_FMT in the future we may need to use
#            D=NINT_WRITE_UNMATCH.
# 6/28/96  - *kob*  modify linux entry for an actual linux machine
#		    with fortran
# 11/12/96 - *kob* - modify makefile so by default it will create both a 
#		   - non-debug and debug version of the tmap libray
# 11/14/96 - *kob* - Add Linux entry similar to that of AIX - linux f77
#		     doesn't use pre-processor
# 08/19/97 - *kob* - added "-Dreclen_in_bytes" to aix flags

#
# Site-specific defines
#
include ../../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../../platform_specific.mk.$(HOSTTYPE)

include ./SOURCE_FILES
include ./LIB_NAME

OBJS= $(SRCS_F:.F=.o) $(SRCS_C:.c=.o) 

$(LIB): $(OBJS) 
	ar r $(LIB) $(OBJS)
	$(RANLIB) $(LIB)

clean:
	rm -f *.o *.f $(LIB)

