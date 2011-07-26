#
# Makefile for gksm2ps with xpreview *jd* 9.15.95
#

include ../site_specific.mk

all:
	${MAKE} $(BUILDTYPE)

i386-linux:	
	$(MAKE) gksm2ps \
		CC=gcc \
		CFLAGS="-m32 -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lX11 -lm"

i486-linux:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-m32 -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lX11 -lm"

x86_64-linux:	
	$(MAKE) gksm2ps \
		CC=gcc \
		CFLAGS="-m64 -I../xgks/port -I../xgks/src/lib -I../xgks/src/lib/gksm" \
		LDFLAGS= \
		GKLIB="../xgks/src/lib/libxgks.a" \
		LIB="-lX11 -lm"

gksm2ps:
	$(CC) $(CFLAGS) $(LDFLAGS) gksm2ps.c xpreview.c \
	 resize_xgks_window.c set_background.c gksm.c batch_graphics.c \
	../ppl/tmapadds/ws_line_bundle.o \
	../ppl/tmapadds/restore_saved_line_colors.o \
	../ppl/tmapadds/query_xgks_wkstn.o \
	$(GKLIB) $(LIB) -o gksm2ps

clean:
	rm -f *.o gksm2ps

#
