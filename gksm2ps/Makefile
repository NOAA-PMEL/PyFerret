#
# Makefile for gksm2ps with xpreview *jd* 9.15.95
#
# Modified for CVS directory structure *jd* 3.26.99

include ../site_specific.mk

all:
	${MAKE} $(BUILDTYPE)

solaris8:	
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I../xgks/port -I../xgks/src/lib -I/usr/openwin/share/include -I../xgks/src/lib/gksm" \
		LDFLAGS="-z muldefs" \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lnsl -L/usr/openwin/lib -lX11 -lm"
solaris10:	
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I../xgks/port -I../xgks/src/lib -I/usr/openwin/share/include -I../xgks/src/lib/gksm" \
		LDFLAGS="-z muldefs" \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lnsl -L/usr/openwin/lib -lX11 -lm"

i386-linux:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-m32 -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lX11 -lm"

i386-apple-darwin:
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-I../xgks/port -I../xgks/src/lib -I/usr/openwin/share/include -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-L/usr/X11R6/lib -lX11 -lm"

i486-linux:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-m32 -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lX11 -lm"

i586:
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-I/usr/lib/gcc-lib/i686-pc-cygwin/2.95.3-5/include/ 
			  -I/usr/include \
			  -I/usr/include/X11 \
			  -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-L/usr/X11R6.4/lib -lX11 -lm"

intel-mac:
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm -I/usr/X11/include" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-L/usr/X11/lib -lX11 -lm"

x86_64-linux:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-m64 -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lX11 -lm"

x86_64-linux-gnu:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-m64 -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lX11 -lm"

gksm2ps:
	$(CC) $(CFLAGS) $(LDFLAGS) -o gksm2ps gksm2ps.c xpreview.c \
	 resize_xgks_window.c set_background.c gksm.c batch_graphics.c \
	../ppl/tmapadds/ws_line_bundle.o \
	../ppl/tmapadds/restore_saved_line_colors.o \
	../ppl/tmapadds/query_xgks_wkstn.o \
	$(GKLIB) $(LIB)
	chmod +x gksm2ps

clean:
	rm -f *.o gksm2ps

