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
F77 = f77
CC = cc
RANLIB=ranlib
DEBUG=

FFILES=*.F
CFILES=*.c

OBJS=ez*.o gt*.o lib_*.o str_*.o tm_*.o mc*.o ts_*.o x*data.o cd*.o \
	readline.o \
	history.o \
	keymaps.o \
	funmap.o \
	nap.o \
	free_time.o \
	alloca.o

all_systems:
	${MAKE} $(HOSTTYPE)_comp
	${MAKE} $(HOSTTYPE)
	${MAKE} $(HOSTTYPE)_gcomp
	${MAKE} $(HOSTTYPE)_deblib
	
sgi:
	${MAKE} tmap_libs/tmap_lib.a FFLAGS="-c -Nl60 -w -G 0 -Dsgi -Dsun" CCFLAGS="-cckr -Ireadline -I. -I/usr/local/include  -c -G 0 -Dsgi -Dsun -D_BSD_SIGNALS" RANLIB="ar ts"

sun4:  
	${MAKE} tmap_libs/tmap_lib.a FFLAGS="-c -Nl60 -w -u" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -D_NO_PROTO -c"

solaris:
	${MAKE} tmap_libs/tmap_lib.a FFLAGS="-c  -Dreclen_in_bytes -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Nl60 -w -u" CCFLAGS="-Dsolaris -DUSG -I/opt/include -I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c" RANLIB="touch" 

decstation:
	${MAKE} tmap_libs/tmap_lib.a F77="f77" FFLAGS="-c -u -cpp -w1 -G 0 -fpe1 -check underflow -check overflow" CCFLAGS="-Ireadline -I. -I/usr/local/include -c"

alpha:
	${MAKE} tmap_libs/tmap_lib.a F77="f77" FFLAGS="-c -u -cpp -w1 -align dcommons -align records -integer_size 32 -real_size 32  -check overflow -check underflow" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -DUSG -DDEC_ALPHA"	DEBUG=" "

aix:
	${MAKE} tmap_libs/tmap_lib.a F77="xlf" FFLAGS="-c -u -qfixed=132" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -Dreclen_in_bytes -DUSG -Dunix -cpp -DVOID_SIGHANDLER -Daix -Daix_cc -DNO_ENTRY_NAME_UNDERSCORES -Dsun"	


hp:  
	${MAKE} tmap_libs/tmap_lib.a FFLAGS="-c -C +es +e -u -w -Dsun -DREADONLY_ALLOWED -Dreclen_in_bytes -DNEED_IAND +U77" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c -D_NO_PROTO -DNO_ENTRY_NAME_UNDERSCORES -DHP_CC -Dsun"

linux:
	${MAKE} tmap_libs/tmap_lib.a F77="f90" FFLAGS="-132 -c -dusty -C -f77 -fieee -Nl60 -x77 -w" CPPFLAGS="-P -Dunix -DFORTRAN_90 -DNEED_IAND" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -Dunix -Dsolaris -DVOID_SIGHANDLER -D_POSIX_VERSION -DNEED_IAND" DEBUG=" "
	

tmap_libs/tmap_lib.a:	$(OBJS)
		- mv tmap_libs/tmap_lib.a tmap_libs/tmap_lib.a.pre_`date +%m%d%y`
		ar r tmap_libs/tmap_lib.a $(OBJS)
		$(RANLIB) tmap_libs/tmap_lib.a


deblib:
	${MAKE} $(HOSTTYPE)_deblib
	
sgi_deblib:
	${MAKE} tmap_libs/tmap_deblib.a FFLAGS="-c -Nl60 -w -G 0 -Dsgi -Dsun" CCFLAGS="-cckr -Ireadline -I. -I/usr/local/include -c -G 0 -Dsgi -Dsun -D_BSD_SIGNALS" RANLIB="ar ts" DEBUG="-g"

sun4_deblib:
	${MAKE} tmap_libs/tmap_deblib.a FFLAGS="-c -Nl60 -w -u" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c -D_NO_PROTO" DEBUG="-g"

solaris_deblib:
	${MAKE} tmap_libs/tmap_deblib.a FFLAGS="-c  -Dreclen_in_bytes -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Nl60 -w -u" CCFLAGS="-Dsolaris -DUSG _/opt/include -I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c" RANLIB="touch" DEBUG="-g"

decstation_deblib:
	${MAKE} tmap_libs/tmap_deblib.a F77="f77" FFLAGS="-c -u -cpp -w1 -G 0 -fpe1 -check underflow -check overflow" CCFLAGS="-Ireadline -I. -I/usr/local/include -c" DEBUG="-g"

alpha_deblib:
	${MAKE} tmap_libs/tmap_deblib.a F77="f77" FFLAGS="-c -u -cpp -w1 -align dcommons -align records -integer_size 32 -real_size 32  -check overflow -check underflow" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -DUSG -DDEC_ALPHA"	DEBUG="-g"

aix_deblib:
	${MAKE} tmap_libs/tmap_deblib.a F77="xlf" FFLAGS="-g -c -u -qfixed=132" CCFLAGS="-Ireadline -I. -I/usr/local/include -g -c -Dreclen_in_bytes -DUSG -Dunix -cpp -DVOID_SIGHANDLER -Daix_cc -DNO_ENTRY_NAME_UNDERSCORES -Dsun" CPPFLAGS="-Dsun -Dunix -Daix -P -DAIX_XLF -DNO_ENTRY_NAME_UNDERSCORES"	

hp_deblib:
	${MAKE} tmap_libs/tmap_deblib.a  FFLAGS="-c -C +es +e -u -w -Dsun -DREADONLY_ALLOWED -Dreclen_in_bytes -DNEED_IAND +U77" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c -D_NO_PROTO -DNO_ENTRY_NAME_UNDERSCORES -DHP_CC -Dsun" DEBUG="-g"

linux_deblib:
	${MAKE} tmap_libs/tmap_deblib.a F77="f90" FFLAGS="-132 -c -dusty -C -f77 -fieee -g -Nl60 -x77 -w" CPPFLAGS="-P -Dunix -DNEED_IAND -DFORTRAN_90 -DNEED_IAND" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -Dunix -Dsolaris -DVOID_SIGHANDLER -D_POSIX_VERSION -DNEED_IAND" DEBUG="-g"






tmap_libs/tmap_deblib.a:	$(OBJS)
		- mv tmap_libs/tmap_debuglib.a tmap_libs/tmap_debuglib.a.pre_`date +%m%d%y`
		ar r tmap_libs/tmap_debuglib.a $(OBJS)
		$(RANLIB) tmap_libs/tmap_debuglib.a

	

links:
	/bin/rm -f tmap_libs
	/bin/rm -f tmap_format
	ln -s ../cmn          tmap_format
	ln -s ../libs         tmap_libs

.F.o: 
	($(F77) $(DEBUG) $(FFLAGS) $(<F))

.c.o:
	(cc $(CCFLAGS) -g $(<F))


clean:
	/bin/rm *.o

nogcompile:
	${MAKE} $(HOSTTYPE)_comp 

	
sgi_comp:
	${MAKE} comp FFLAGS="-c -Nl60 -w -G 0 -Dsgi -Dsun" CCFLAGS="-cckr -Ireadline -I. -I/usr/local/include -c -G 0 -Dsgi -Dsun -D_BSD_SIGNALS -I/home/e2/tmap/local/sgi/include" 

sun4_comp:  
	${MAKE} comp CC="/usr/ucb/cc"  FFLAGS="-c -Nl60 -w -u" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -D_NO_PROTO -c" 

solaris_comp:  
	${MAKE} comp CC="cc"  FFLAGS="-c  -Dreclen_in_bytes -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Nl60 -w -u" CCFLAGS="-Dsolaris -DUSG -I/opt/include -I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c" 

decstation_comp:
	${MAKE} comp F77="f77" FFLAGS="-c -u -cpp -w1 -G 0 -fpe1 -check underflow -check overflow" CCFLAGS="-Ireadline -I. -I/usr/local/include -c"

alpha_comp:
	${MAKE} comp F77="f77" FFLAGS="-c -u -cpp -C -w1 -align dcommons -align records -integer_size 32 -real_size 32 -check underflow -check overflow -Df7732_bug -DDEC_FORTRAN  " CCFLAGS="-Ireadline -I. -I/usr/local/include -c -DUSG -DDEC_ALPHA"
	
aix_comp:
	${MAKE} comp_nocpp F77="xlf" FFLAGS="-c -u -qfixed=132" CCFLAGS="-Ireadline -I. -I/usr/local/include -cpp -c -DUSG -Dunix -Dreclen_in_bytes -Daix -DVOID_SIGHANDLER -Daix_cc -DNO_ENTRY_NAME_UNDERSCORES -Dsun" CPPFLAGS="-Dsun -Dunix -Daix -P -DAIX_XLF -DNO_ENTRY_NAME_UNDERSCORES -Dreclen_in_bytes"	

hp_comp:  
	${MAKE} comp FFLAGS="-c -C +es +e -u -w -Dsun -DREADONLY_ALLOWED -Dreclen_in_bytes -DNEED_IAND +U77" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c -D_NO_PROTO -DNO_ENTRY_NAME_UNDERSCORES -DHP_CC -Dsun"

linux_comp:
	${MAKE} comp_nocpp F77="f90" FFLAGS="-132 -c -dusty -C -f77 -fieee -Nl60 -x77 -w" CPPFLAGS="-P -Dunix -DNEED_IAND -DFORTRAN_90 -DF90_SYSTEM_ERROR_CALLS -Dreclen_in_bytes" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -Dunix -Dsolaris -DVOID_SIGHANDLER -D_POSIX_VERSION -DNEED_IAND"
	

gcompile:
	${MAKE} $(HOSTTYPE)_gcomp

sgi_gcomp:
	${MAKE} comp FFLAGS="-c -Nl60 -w -G 0 -Dsgi -Dsun" CCFLAGS="-cckr -Ireadline -I. -I/usr/local/include -c -G 0 -Dsgi -Dsun -D_BSD_SIGNALS" DEBUG="-g"

sun4_gcomp:  
	${MAKE} comp CC="/usr/ucb/cc"  FFLAGS="-c -Nl60 -w -u" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -D_NO_PROTO -c" DEBUG="-g"

solaris_gcomp:  
	${MAKE} comp CC="cc"  FFLAGS="-c  -Dreclen_in_bytes -DNEED_BYTE -DACCEPTS_BYTE_DECLARATION -Nl60 -w -u" CCFLAGS="-Dsolaris  -DUSG -I/opt/include -I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c" DEBUG="-g"

decstation_gcomp:
	${MAKE} comp F77="f77" FFLAGS="-c -u -cpp -w1 -G 0 -fpe1 -check underflow -check overflow" CCFLAGS="-Ireadline -I. -I/usr/local/include -c" DEBUG="-g"

alpha_gcomp:
	${MAKE} comp F77="f77" FFLAGS="-c -u -cpp -C -w1 -align dcommons -align records -integer_size 32 -real_size 32 -check underflow -check overflow" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -DUSG -DDEC_ALPHA" DEBUG="-g"
	
aix_gcomp:
	${MAKE} comp_nocpp F77="xlf" FFLAGS="-c -u -qfixed=132" DEBUG="-g" CCFLAGS="-Ireadline -I. -I/usr/local/include -cpp -c -g -DUSG -Dreclen_in_bytes -Dunix -Daix -DVOID_SIGHANDLER -Daix_cc -DNO_ENTRY_NAME_UNDERSCORES -Dsun" CPPFLAGS=" -Dsun -Dunix -Daix -P -DAIX_XLF -Dreclen_in_bytes"

hp_gcomp:  
	${MAKE} comp FFLAGS="-c -C +es +e -u -w -Dsun -DREADONLY_ALLOWED -Dreclen_in_bytes -DNEED_IAND +U77" CCFLAGS="-I/usr/ucbinclude -Ireadline -I. -I/usr/local/include -c -D_NO_PROTO -DNO_ENTRY_NAME_UNDERSCORES -DHP_CC -Dsun" DEBUG="-g"

linux_gcomp:
	${MAKE} comp_nocpp F77="f90" FFLAGS="-132 -c -dusty -C -f77 -fieee -Nl60 -x77 -w" CPPFLAGS="-P -Dunix -DNEED_IAND -DFORTRAN_90 -DF90_SYSTEM_ERROR_CALLS -Dreclen_in_bytes" CCFLAGS="-Ireadline -I. -I/usr/local/include -c -Dunix -Dsolaris -DVOID_SIGHANDLER -D_POSIX_VERSION -DNEED_IAND" DEBUG="-g"
	


comp:
	$(F77) $(DEBUG) $(FFLAGS) *.F
	$(CC) $(CCFLAGS) $(DEBUG) *.c


comp_nocpp:
	for file in ${FFILES} ; \
	do \
		touch `echo $${file} | sed -e 's/.F/.f/g'`; \
		rm `echo $${file} | sed -e 's/.F/.f/g'`; \
                /lib/cpp ${CPPFLAGS} $${file} > `echo $${file} | sed -e 's/.F/.f/g'`; \
                ${F77} $(FFLAGS) $(DEBUG) `echo $${file} | sed -e 's/.F/.f/g'`; \
	done
	$(CC) $(CCFLAGS) $(DEBUG) *.c



