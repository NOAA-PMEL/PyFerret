#
#	Makefile for pplp
#	This version used on all systems
#	J Davison 2.93
#
#	Mod to use f77 (3.2) compiler *jd* 3.16.93
#	 - Change F77 TMLIB GKLIB and XLIB
#	Mod to include XGKS with GK=xgks argument (make GK=xgks) *jd* 1.28.94
#	      -make xgks by default		
#	Mod to permit passing G=-g to compile debug
#	Mod to work on AIX *jd* 10.17.94
# 	Mod to work on Linux *kob* 5.15.97
GK=xgks

DIRS = addsxgks checkdir complot modsxgks ourlib pplepic pplp_only pplusr symlib plot plotlib

all_systems:
	${MAKE} $(HOSTTYPE)$(GK)
decstation:
	${MAKE} pplp \
		F77=f77 \
		CC=cc \
		LDFLAGS= \
		FFLAGS="-fpe1 $(G) -c -static -G 0 -O0" \
		CFLAGS="$(G) -c" \
		GKOBJ=GKS3Dconfig.o \
		GKOBJ2=GKS3Dconfig_motif.o \
		GKLIB="-lGKS3Dmotif -lGKS3Ddforbnd -lGKS3D" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-lMrm -lXm -lXt -ldwt -lcursesX -li -lc -lX11" 
sun4:
	${MAKE} pplp \
		F77=f77 \
		CC=cc \
		LDFLAGS= \
		FFLAGS="$(G) -c -Nl160 -Datc_gks -static" \
		CFLAGS="$(G) -c" \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(gksdir) -lgksflb -lgksgksm -lgksw5300 -lgksw1900 -lgksw2500 -lgksw3100 -lgksw10100 -lgksmsc" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-lX11"
alpha:	
	${MAKE} pplp \
		F77=f77 \
		CC=cc \
		LDFLAGS="-non_shared" \
		FFLAGS="$(G) -c -Nl160 -Datc_gks -static -cpp\
		-integer_size 32 -real_size 32 -align dcommons -align records"\
		CFLAGS="$(G) -c" \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(gksdir) -lgksflb -lgksgksm -lgksw5300 -lgksw1900 -lgksw2500 -lgksw3100 -lgksw10100 -lgksmsc" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-lX11 -ldnet_stub"
sgi:
	echo "This needs to be modified at SGI site"
	${MAKE} pplp \
		F77=f77 \
		CC=cc \
		LDFLAGS= \
		FFLAGS="($G) -c -Nl160 -Datc_gks -static -G 0" \
		CFLAGS="($G) -c" \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(gksdir) -lgksflb -lgksgksm -lgksw5300 -lgksw1900 -lgksw2500 -lgksw3100 -lgksw10100 -lgksmsc" \
		RANLIB="/bin/touch" \
		XLIB="-lX11"
decstationxgks:
	${MAKE} pplp \
		F77=f77 \
		LDFLAGS= \
		FFLAGS="-fpe1 $(G) -c -static -G 0 -O0 -Dxgks" \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G)" \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-lX11 -lm" 
sun4xgks:	
	${MAKE} pplp \
		F77=f77 \
		CC=cc \
		LDFLAGS= \
		FFLAGS="$(G) -c -Nl160 -static -Dxgks" \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G)" \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-lX11"
solarisxgks:	
	${MAKE} pplp \
		F77=f77 \
		CC=/opt/SUNWspro/bin/cc \
		LDFLAGS= \
		FFLAGS="$(G) -c -Nl160 -static -Dxgks -DNO_OPEN_READONLY -DNO_OPEN_CARRIAGECONTROL" \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -I/usr/openwin/share/include -c $(G)" \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		RANLIB="/bin/touch" \
		XLIB="-lnsl -L/usr/openwin/lib -lX11"
alphaxgks:	
	${MAKE} pplp \
		F77=f77 \
		CC=cc \
		LDFLAGS="-non_shared" \
		FFLAGS="$(G) -c -Nl160 -Dxgks -static -cpp \
		-integer_size 32 -real_size 32 -align dcommons -align records"\
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G)" \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks$(G)" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-lX11 -ldnet_stub"
sgixgks:	
	${MAKE} pplp \
                F77=f77 \
                CC=cc \
                LDFLAGS= \
                FFLAGS="$(G) -c -Nl160 -Dxgks -static -G 0" \
                CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c ($G)" \
                GKOBJ= \
                GKOBJ2= \
                GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
                RANLIB="/bin/touch" \
                XLIB="-lX11"

hpxgks:
	${MAKE} pplp \
                F77=f77 \
                CC=cc \
                LDFLAGS= \
                FFLAGS="$(G) -c +es +e -w -DNEED_IAND -Dxgks -Dhp -Dunix +U77" \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c ($G) -I/usr/include/X11R5 -DNO_ENTRY_NAME_UNDERSCORES" \
                GKOBJ= \
                GKOBJ2= \
                GKLIB="$(TMAP_LOCAL)/lib/libxgks.a" \
                RANLIB="/bin/touch" \
                XLIB=" /usr/lib/X11R5/libX11.a -lm -lU77"

aixxgks:	
	${MAKE} pplp \
		F77=xlf \
		CC=cc \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G) -DNO_ENTRY_NAME_UNDERSCORES" \
		FFLAGS="$(G) -c -static" \
		LDFLAGS= \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-lX11"

i486-linuxxgks:	
	${MAKE} pplp \
		F77=f90 \
		CC=cc \
		CFLAGS="-Ipplinc/xgks -I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c -O2 -m486 $(G)" \
		FFLAGS="$(G) -132 -c -save -dusty -f77 -fieee -Nl90 -x77 -w -mismatch" \
		LDFLAGS= \
		GKOBJ= \
		GKOBJ2= \
		GKLIB="-L$(TMAP_LOCAL)/lib -lxgks" \
		RANLIB="/usr/bin/ranlib" \
		XLIB="-L/usr/X11R6/lib -lX11 -static /usr/lib/libf2c.a /usr/lib/libc.a /usr/lib/libc_p.a /home/l1/kobrien/src/f2c/libF77/libF77.a /home/l1/kobrien/src/f2c/libI77/libI77.a /usr/local/lib/f90/libf90.a -lm"

update:	
	${MAKE} update_date DATE=`/bin/date +%D`

update_date:
	sed "s&pplpdate&$(DATE)&" < adds/ppladd_pplp_ver_data.template > adds/ppladd_pplp_ver_data.F

OBJ    = checkdir/*.o pplp_only/*.o adds$(GK)/*.o mods$(GK)/*.o 
ARK    = plot/plt.a symlib/sym.a plotlib/pll.a complot/cmp.a \
	 ourlib/our.a pplepic/epi.a pplusr/usr.a

###TMLIB  = $(FER_LINK_DIR)/fmt/libs/tmap_deblib.a 
TMLIB  = $(FER_LINK_DIR)/fmt/libs/tmap_lib.a 

CMPOBJ = complot/*.o
OUROBJ = ourlib/*.o
PLTOBJ = plot/*.o
PLLOBJ = plotlib/*.o
EPIOBJ = pplepic/*.o
SYMOBJ = symlib/*.o
USROBJ = pplusr/*.o

pplp:	$(OBJ) $(ARK) $(GKOBJ) $(GKOBJ2)
	- mv pplp$(GK) pplp$(GK)_pre_`date +'%d%h'|tr '[A-Z]' '[a-z]'`
	- $(F77) $(LDFLAGS) -o pplp$(GK) $(OBJ) $(GKOBJ2) $(ARK) $(TMLIB) $(GKLIB) $(XLIB) 

complot/cmp.a:	$(CMPOBJ)
	(cd complot; /bin/rm -f cmp.a; ar r cmp.a *.o; $(RANLIB) cmp.a)
ourlib/our.a:	$(OUROBJ)
	(cd ourlib;  /bin/rm -f our.a; ar r our.a *.o; $(RANLIB) our.a)
plot/plt.a:	$(PLTOBJ)
	(cd plot;    /bin/rm -f plt.a; ar r plt.a *.o; $(RANLIB) plt.a)
plotlib/pll.a:	$(PLLOBJ)
	(cd plotlib; /bin/rm -f pll.a; ar r pll.a *.o; $(RANLIB) pll.a)
pplepic/epi.a:	$(EPIOBJ)
	(cd pplepic; /bin/rm -f epi.a; ar r epi.a *.o; $(RANLIB) epi.a)
symlib/sym.a:	$(SYMOBJ)
	(cd symlib;  /bin/rm -f sym.a; ar r sym.a *.o; $(RANLIB) sym.a)
pplusr/usr.a:	$(USROBJ)
	(cd pplusr;  /bin/rm -f usr.a; ar r usr.a *.o; $(RANLIB) usr.a)

.F.o:
	(cd $(<D); $(F77) $(FFLAGS) $(<F))
.f.o:
	(cd $(<D); $(F77) $(FFLAGS) $(<F))
.c.o:
	(cd $(<D); $(CC) $(CFLAGS) $(<F))

compile:
	${MAKE} $(HOSTTYPE)$(GK)_compile

gcompile:
	${MAKE} $(HOSTTYPE)$(GK)_compile G=-g

decstation_compile:
	${MAKE} do_the_compile \
		F77=f77 \
		FFLAGS="-fpe1 $(G) -c -static -G 0 -O0"
sun4_compile:	
	${MAKE} do_the_compile \
		F77=f77 \
		CC=cc \
		FFLAGS="$(G) -c -Nl160 -Datc_gks -static" \
		CFLAGS="$(G) -c" 
alpha_compile:	
	${MAKE} do_the_compile \
		F77=f77 \
		CC=cc \
		FFLAGS="$(G) -c -Nl160 -static -cpp -Dxgks \
		-integer_size 32 -real_size 32 -align dcommons -align records" \
		 CFLAGS="$(G) -c"
sgi_compile:
	echo "This needs to be modified at SGI site"
	${MAKE} do_the_compile \
		F77=f77 \
		FFLAGS="$(G) -c -Nl160 -Datc_gks -static -G 0" 
decstationxgks_compile:
	${MAKE} do_the_compile \
		F77=f77 \
		FFLAGS="-fpe1 $(G) -c -static -G 0 -O0 -Dxgks" \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G)"
sun4xgks_compile:	
	${MAKE} do_the_compile \
		F77=f77 \
                GK=xgks \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G)" \
		FFLAGS="$(G) -c -Nl160 -Dxgks -static" 
solarisxgks_compile:	
	${MAKE} do_the_compile \
		F77=f77 \
		CC=/opt/SUNWspro/bin/cc \
                GK=xgks \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -I/usr/openwin/share/include -c $(G)" \
		FFLAGS="$(G) -c -Nl160 -Dxgks -static -DNO_OPEN_READONLY -DNO_OPEN_CARRIAGECONTROL"

alphaxgks_compile:	
	${MAKE} do_the_compile \
		F77=f77 \
		CC=cc \
		FFLAGS="$(G) -c -Nl160 -Dxgks -static -cpp\
		-integer_size 32 -real_size 32 -align dcommons -align records" \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G)"

sgixgks_compile:
	${MAKE} do_the_compile \
                F77=f77 \
                GK=xgks \
                CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G)" \
                FFLAGS="$(G) -c -Nl160 -Dxgks -static -G 0" 

hpxgks_compile:
	${MAKE} do_the_compile \
                F77=f77 \
		CC=cc \
                GK=xgks \
                CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G) -I/usr/include/X11R5 -DNO_ENTRY_NAME_UNDERSCORES" \
                FFLAGS="$(G) -c +es +e -w -DNEED_IAND -Dxgks -Dhp -Dunix +U77"

aixxgks_compile:	
	${MAKE} do_the_compile_aix \
		F77=xlf \
                GK=xgks \
		CFLAGS="-I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c $(G) -DNO_ENTRY_NAME_UNDERSCORES" \
		FFLAGS="$(G) -c -static" \
		CPPFLAGS="-P -Dunix -DNO_PASSED_CONCAT -DNO_SNGL -DMANDATORY_FORMAT_WIDTHS -DNO_ACCESS_APPEND -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE -DNO_OPEN_READONLY -DNO_OPEN_CARRIAGECONTROL $(GKSFFLAGS) $(MEMFLAGS) -Dxgks -Daix -DAIX_XLF"

i486-linuxxgks_compile:	
	${MAKE} do_the_compile_linux \
		F77=f90 \
                GK=xgks \
		CFLAGS="-Ipplinc/xgks -I$(TMAP_LOCAL)/src/xgks-2.5.5/port -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib -I$(TMAP_LOCAL)/src/xgks-2.5.5/src/lib/gksm -c -O2 -m486 $(G)" \
		FFLAGS="$(G) -132 -c -save -dusty -f77 -fieee -Nl90 -x77 -w -mismatch" \
		CPP_FLAGS="-P -Dunix -DNEED_IAND -DFORTRAN_90 -DNO_OPEN_SHARED -DNO_OPEN_RECORDTYPE -DNO_OPEN_READONLY -DNO_OPEN_CARRIAGECONTROL -Dxgks -DSTAR_1_SUPPORTED -DF90_NO_FPUTC -DF90_NO_UNLINK -DF90_NO_GETLOG -DF90_SYSTEM_ERROR_CALLS -DF90_DATE_TIME"
     
do_the_compile:
	(cd pplp_only;	$(F77) $(FFLAGS) *.[Ff])
	(cd adds$(GK);  $(F77) $(FFLAGS) *.[Ff])
	(cd adds$(GK);  $(CC)  $(CFLAGS) *.c)
	(cd mods$(GK);  $(F77) $(FFLAGS) *.[Ff])
	(cd mods$(GK);  $(CC)  $(CFLAGS) *.c)
	(cd checkdir; 	$(F77) $(FFLAGS) *.[Ff])
	(cd complot; 	$(F77) $(FFLAGS) *.[Ff])
	(cd ourlib;  	$(F77) $(FFLAGS) *.[Ff])
	(cd plot;    	$(F77) $(FFLAGS) *.[Ff])
	(cd plotlib; 	$(F77) $(FFLAGS) *.[Ff])
	(cd pplepic; 	$(F77) $(FFLAGS) *.[Ff])
	(cd pplusr;  	$(F77) $(FFLAGS) *.[Ff])
	(cd symlib;  	$(F77) $(FFLAGS) *.[Ff])

do_the_compile_aix:
	for dir in ${DIRS} ; \
	do \
		echo $${dir}; cd $${dir}; \
			for file in *.F ; do \
                		rm -f `echo $$file | sed -e 's/.F/.f/g'`; \
				echo $$file;\
		           	sed -e 's/???/?/g' $$file |sed -e 's/??/?/g' |	        /usr/lib/cpp ${CPPFLAGS}    | sed -e 's/	/      /'|sed -e 's/, /,/g'|sed -e 's/\.	/\./g' | sed -e 's/( /(/g' | sed -e 's/ )/)/g' | sed -e 's/\.      /\./g' | sed -e 's/          /      /g' | sed -e 's/        /      /g' | sed -e 's/) /)/g' | sed -e 's/ (/(/g' | sed -e 's/??/?/g' > `echo $$file | sed -e 's/.F/.f/g'`; ${F77} ${FFLAGS} `echo $$file | sed -e 's/.F/.f/g'`; \
			done;\
		cd ..; echo `pwd`; \
	done

	(cd adds$(GK);  $(CC)  $(CFLAGS) *.c)
	(cd mods$(GK);  $(CC)  $(CFLAGS) *.c)

do_the_compile_linux:
	for dir in ${DIRS} ; \
	do \
		echo $${dir}; cd $${dir}; \
			for file in *.F ; do \
                		rm -f `echo $$file | sed -e 's/.F/.f/g'`; \
				echo $$file; \
				/lib/cpp ${CPP_FLAGS} $$file      | sed -e 's/de	/de /g' | sed -e  's/de 	/de /g'> `echo $$file | sed -e 's/.F/.f/g'`; ${F77} ${FFLAGS} `echo $$file | sed -e 's/.F/.f/g'`; \
			done;\
		cd ..; echo `pwd`; \
	done

	(cd adds$(GK);  $(CC)  $(CFLAGS) *.c)
	(cd mods$(GK);  $(CC)  $(CFLAGS) *.c)

libraries:
	make complot/cmp.a
	make ourlib/our.a
	make plot/plt.a
	make plotlib/pll.a
	make pplepic/epi.a
	make symlib/sym.a
	make pplusr/usr.a






