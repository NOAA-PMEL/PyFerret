#
# Makefile (for the FERRET/fer code)
#
# August 11 1997
# Jonathan Callahan (with some help from Joe Sirott)
#
# 10/97 *kob*  - pltform specific files will reside in ppl directory and be 
#		 links to actual files in fer directory
# 7/98 *kob* - reworking prior to source code distribution. add in 
#	       fmt are as well so one make command compiles/links all

#
# include platform specific macro definitions
#

include platform_specific_flags.mk.$(HOSTTYPE)

CFLAGS += -I../gui
#
# Macros
#

#*kob* 2/99 - set GUI_FAKES so that by default it is assumed no gui
#	      stuff is needed.  in order to allow the "make" command 
#	      by itself to work.

GUI_FAKES = special/gui_fakes.o

DIRS	= ccr ctx dat doo efn fmt gnl gui ino mem ocn plt \
	rpn special stk utl xeq

LIBS	= ccr ctx dat doo efn fmt gnl gui ino mem ocn plt \
	rpn stk utl xeq

CLEANDIRS = $(DIRS)

DEPDIRS = ccr gui

FERLIBS = -L../lib -lxeq -lgnl -lferplt -lrpn -lstk -ldoo -locn \
	-lctx -lfmt -lino -lmem -lutl -ldat -lccr -lefn 

PPLLIBS = -L../lib -lplt -lpll -lsym -lcmp -lour -lepi -lusr 

FERMAIN = ccr/fermain_c.o ccr/gui_init.o

FEROBJS_DFLT = $(LINUX_OBJS) \
	special/ferret_dispatch.o special/xmake_date_data.o special/fakes3.o \
	special/ferret_query_f.o  special/xrevision_type_data.o

FEROBJS = ccr/save_arg_pointers.o $(FEROBJS_DFLT)

PPLOBJS =  ../ppl/tmapadds/*.o 

EXECNAME = ferret_c

WHATGKS = xgks

LD = cc

#
# Targets
#

all:	fmt_libraries fer_libraries ppl_libraries special_code ferret


debug:	.FRC
	(cd ccr; $(CC) -c -g $(CFLAGS) gui_init.c)
	$(MAKE) "DEBUG=-g" "DEBUG_TITLE=debug" "FFLAGS = $(FFLAGS) -Ddebug" "GUI_FAKES=special/gui_fakes.o" all

gui_nodebug: .FRC
	(cd ccr; $(CC) -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) "CFLAGS = $(CFLAGS) -DLINK_GUI_AS_MAIN" \
		"GUILIB = -L../lib -lgui $(TMAP_LOCAL)/lib/liblist.a" \
		"MOVIELIB=$(TMAP_LOCAL)/lib/libXpm.a" "EXECNAME=ferretgui_c" \
		"GUI=GUI" all
gui_debug: .FRC
	(cd ccr; $(CC) -g -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) "CFLAGS = $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG=-g"\
		"GUILIB = -L../lib -lgui $(TMAP_LOCAL)/lib/liblist.a" \
		"MOVIELIB=$(TMAP_LOCAL)/lib/libXpm.a" "EXECNAME=ferretgui_c" \
		"FFLAGS = $(FFLAGS) -Ddebug" "GUI=GUI" "DEBUG_TITLE=debug" \
		"GUI_FAKES="  all

DODS_gui_debug: .FRC
	(cd ccr; $(CC) -g -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) "CFLAGS = $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG=-g"\
		"EXECNAME=ferretdods_gui" \
		"GUILIB = -L../lib -lgui $(TMAP_LOCAL)/lib/liblist.a" \
		"MOVIELIB=$(TMAP_LOCAL)/lib/libXpm.a"  \
		"LD = g++" "CDFLIB=-L$(TMAP_LOCAL)/src/DODS/lib -lnc-dods -ldap++ -lnc-dods -ldap++ -lwww -lstdc++ -lrx -lz -lexpect5.21 -ltcl7.6" \
		"FFLAGS = $(FFLAGS) -Ddebug" "GUI_FAKES=" \
		"DEBUG_TITLE=debug" all
DODS_gui: .FRC
	(cd ccr; $(CC) -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) "CFLAGS = $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG="\
		"EXECNAME=ferretdods_gui" \
		"GUILIB = -L../lib -lgui $(TMAP_LOCAL)/lib/liblist.a" \
		"MOVIELIB=$(TMAP_LOCAL)/lib/libXpm.a"  \
		"LD = g++" "CDFLIB=-L$(TMAP_LOCAL)/src/DODS/lib -lnc-dods -ldap++ -lnc-dods -ldap++ -lwww -lstdc++ -lrx -lz -lexpect5.21 -ltcl7.6" \
		"FFLAGS = $(FFLAGS)" "GUI_FAKES=" \
		"DEBUG_TITLE=" all

fortran: .FRC
	$(MAKE) "FERMAIN=special/fermain_fortran.o" "EXECNAME=ferret_fortran" all


ferret:	update
	- mv $(EXECNAME) $(EXECNAME)_pre_`date +'%d%h'|tr '[A-Z]' '[a-z]'`
	- $(LD) $(LDFLAGS) -o $(EXECNAME) \
		$(FERMAIN) \
		$(FEROBJS) \
		$(GUI_FAKES) \
		$(XTRA_OBJ) \
		$(GUILIB) \
		$(GKSOBJ) \
		$(PPLOBJS) \
		$(BDEAD) \
		$(FERLIBS) \
		$(PPLLIBS) \
		$(MOVIELIB) \
		$(TMLIB) \
		-ltermcap \
		$(CDFLIB) \
		$(HDFLIB) \
		$(GKSLIB) \
		$(SYSLIB) \
		$(DODS_SYSLIB) 
	chmod +x $(EXECNAME)

update: .FRC
	${MAKE} update_date DATE=`/bin/date +%D`
	sed -e 's&mmmmmmmm&($(DEBUG_TITLE)/$(GUI))&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	(cd special;make)

update_date:
	sed "s&xxxxxxxx&$(DATE)&" <special/xmake_date_data.template >special/xmake_date_data.F


libraries: ppl_libraries fer_libraries


fmt_libraries: .FRC
	(cd ../fmt/src ; $(MAKE));

fer_libraries: .FRC
	for i in $(LIBS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; \
        $(MAKE) "MFLAGS=$(MFLAGS)" "CC=$(CC)" "CFLAGS=$(DEBUG) $(CFLAGS)" \
                "FFLAGS=$(DEBUG) $(FFLAGS)" "PPLUS_FFLAGS=$(PPLUS_FFLAGS)"); \
        done


ppl_libraries: .FRC
	(cd ../ppl ; $(MAKE) "PPLUS_FFLAGS=$(DEBUG) $(PPLUS_FFLAGS)" all);


special_code: .FRC
	(cd special ; $(MAKE) all);


clean: fmt_clean ppl_clean fer_clean 
	(cd ../lib ; rm -f *.a *.so);

ppl_clean: .FRC
	(cd ../ppl ; $(MAKE) clean);

fer_clean: .FRC
	for i in $(CLEANDIRS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; $(MAKE) clean); \
        done

fmt_clean: .FRC
	(cd ../fmt/src; rm -f *.[fo])

depend: .FRC
	(cd ../ppl ; $(MAKE) depend);
	for i in $(DEPDIRS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; \
        $(MAKE) "MFLAGS=$(MFLAGS)" "CC=$(CC)" "CFLAGS=$(CFLAGS)" \
                "FFLAGS=$(FFLAGS)" "PPLUS_FFLAGS=$(PPLUS_FFLAGS)" $@); \
        done


.FRC:

#
# End of Makefile
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
