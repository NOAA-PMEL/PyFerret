#
# Makefile for gksm2ps with xpreview *jd* 9.15.95
#

all_systems:
	${MAKE} $(HOSTTYPE)

decstation:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"
sun4:	
	${MAKE} gksm2ps \
		CC=gcc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"
alpha:	
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm $(G)" \
		LDFLAGS="-non_shared" \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks$(G) " \
		LIB="-lX11 -lm -ldnet_stub"

solaris:	
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I/usr/openwin/share/include -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lnsl -L/usr/openwin/lib -lX11 -lm"
sgi:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"

aix:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="-lX11 -lm"

hp:
	${MAKE} gksm2ps \
		CC=cc \
		CFLAGS=" -I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I/usr/include/X11R5 -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm $(G)" \
		LDFLAGS= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		LIB="/usr/lib/X11R5/libX11.a -lm"


gksm2ps:	gksm2ps.o xpreview.o \
	resize_xgks_window.o set_background.o gksm.o batch_graphics.o
	- $(CC) $(CFLAGS) $(LDFLAGS) -o gksm2ps gksm2ps.o xpreview.o \
	 resize_xgks_window.o set_background.o gksm.o batch_graphics.o \
	../ppl/addsxgks/ppladd_ws_line_bundle.o \
	../ppl/addsxgks/ppladd_query_xgks_wkstn.o \
	../ppl/modsxgks/xgksmod_xopws.o \
	../ppl/modsxgks/xgksmod_xcolours.o \
	$(GKLIB) $(LIB)
	chmod +x gksm2ps

clean:
	rm *.o gksm2ps


