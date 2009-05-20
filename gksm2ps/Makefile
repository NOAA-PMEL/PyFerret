#
# Makefile for gksm2ps with xpreview *jd* 9.15.95
#
# Modified for CVS directory structure *jd* 3.26.99


all_systems:
	${MAKE} $(HOSTTYPE)

decstation:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks/port -I$(TMAP_LOCAL)/src/xgks/src/lib -I$(TMAP_LOCAL)/src/xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"
sun4:
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks/port -I$(TMAP_LOCAL)/src/xgks/src/lib -I$(TMAP_LOCAL)/src/xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"
alpha:	
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks/port -I$(TMAP_LOCAL)/src/xgks/src/lib -I$(TMAP_LOCAL)/src/xgks/src/lib/gksm $(G)" \
		LDFLAGS="-non_shared" \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks$(G) " \
		LIB="-lX11 -lm -ldnet_stub"

solaris:	
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks/port -I$(TMAP_LOCAL)/src/xgks/src/lib -I/usr/openwin/share/include -I$(TMAP_LOCAL)/src/xgks/src/lib/gksm $(G)" \
		LDFLAGS="-z muldefs" \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lnsl -L/usr/openwin/lib -lX11 -lm"
solaris8:	
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I../xgks/port -I../xgks/src/lib -I/usr/openwin/share/include -I../xgks/src/lib/gksm $(G)" \
		LDFLAGS="-z muldefs" \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lnsl -L/usr/openwin/lib -lX11 -lm"
iris4d:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks/port -I$(TMAP_LOCAL)/src/xgks/src/lib -I$(TMAP_LOCAL)/src/xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"

aix:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks/port -I$(TMAP_LOCAL)/src/xgks/src/lib -I$(TMAP_LOCAL)/src/xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"

hp:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS=" -I$(TMAP_LOCAL)/src/xgks/port -I$(TMAP_LOCAL)/src/xgks/src/lib -I/usr/include/X11R5 -I$(TMAP_LOCAL)/src/xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="/usr/lib/X11R5/libX11.a -lm"

i386-linux:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-g -I/usr/lib/gcc-lib/i386-redhat-linux/3.2.3/include -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-L/usr/X11R6/lib -lX11 -lm"

i486-linux:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-g -I/usr/lib/gcc-lib/i386-redhat-linux/3.2.3/include -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-L/usr/X11R6/lib -lX11 -lm"

i586:
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-I/usr/lib/gcc-lib/i686-pc-cygwin/2.95.3-5/include/ -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-L/usr/X11R6.4/lib -lX11 -lm"

x86_64-linux:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-g -I/usr/lib/gcc/x86_64-redhat-linux/3.4.3/include -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-L/usr/X11R6/lib64 -lX11 -lm"

gksm2ps:	gksm2ps.o xpreview.o \
	resize_xgks_window.o set_background.o gksm.o batch_graphics.o
	- $(CC) $(CFLAGS) $(LDFLAGS) -o gksm2ps gksm2ps.o xpreview.o \
	 resize_xgks_window.o set_background.o gksm.o batch_graphics.o \
	../ppl/tmapadds/ws_line_bundle.o \
	../ppl/tmapadds/restore_saved_line_colors.o \
	../ppl/tmapadds/query_xgks_wkstn.o \
	$(GKLIB) $(LIB)
	chmod +x gksm2ps


clean:
	rm *.o gksm2ps
