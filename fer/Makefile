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
# 7/99 *sh* - corrected misbehavior of make debug -- make update was not called
# 8/99 *acm* - Add EFUOBJS and ef_utility to DIRS and LIBS so to include 
#              External Functions utility functions in the executable.
#              (see note in ef_utility/Makefile: Don't put these in a library.)
# 12/99 *sh* - changed "debug" to "alpha" as DEBUG_TITLE
#  3/00 *acm* - Add efi library External Functions, internally linked
#  5/00 *kob* - Added a "make update" to all entries in order to properly
#	        update the date for each make.  this was also necessary when
#		compiling Ferret initially in order to create 
#	        special/xmake_date_data.F  and special/xrevision_type_data.F
#  v5.4 10/01*kob* - removed explicit reference to libXpm.a  - move it instead 
#                  to system specific SYS_LIBS macro in platform specific files
#  v5.4 10/01 *kob* - add the removal of .f files - for linux
#  v5.41 4/02 *acm* - Add -g flag for debug compiles of C code.  
# v5.54  8/04 *acm* - Add target "profiling", to compile with -pg option.
# v6.1 beta 3/06 *acm* - For Beta release to GFDL, add (beta) as DEBUG_TITLE
# v6.1 release 3/06 *acm* - Remove (beta) as DEBUG_TITLE for official release
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

DIRS	= ccr ctx dat doo efi efn ef_utility fmt gnl gui ino mem  \
	ocn plt rpn special stk utl xeq

LIBS	= ccr ctx dat doo efi efn ef_utility fmt gnl gui ino mem \
	ocn plt rpn stk utl xeq

CLEANDIRS = $(DIRS)

DEPDIRS = ccr gui

FERLIBS = -L../lib -lxeq -lgnl -lferplt -lrpn -lstk -ldoo -locn \
	-lctx -lfmt -lino -lmem -lutl -ldat -lccr -lefi -lefn

PPLLIBS = -L../lib -lplt -lpll -lsym -lcmp -lour -lepi -lusr 

FERMAIN = ccr/fermain_c.o ccr/gui_init.o

FEROBJS_DFLT = $(LINUX_OBJS) \
	special/ferret_dispatch.o special/xmake_date_data.o special/fakes3.o \
	special/ferret_query_f.o  special/xrevision_type_data.o \
	special/xplatform_type_data.o

FEROBJS = ccr/save_arg_pointers.o $(FEROBJS_DFLT)

PPLOBJS =  ../ppl/tmapadds/*.o 

EFUOBJS = ef_utility/*.o

EFIOBJS = efi/*.o

EXECNAME = ferret_c

WHATGKS = xgks

LD = $(CC) 

EF_CMNS = common/EF_Util.cmn common/EF_Util.parm common/EF_mem_subsc.cmn

#
# Targets
#

all:	fmt_libraries fer_libraries ppl_libraries special_code ferret

# Dont seem to have all the static libraries we need for this.
ncdf4_static:	.FRC
	(cd ccr; $(CC) -c $(CFLAGS) gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = $(CFLAGS) " "DEBUG="\
                "LD = g++" "LDFLAGS=-v --verbose -L/lib -L/usr/X11R6/lib64  -static" \
		"FFLAGS = $(FFLAGS)"  \
		all

ncdf4:	.FRC
	(cd ccr; $(CC) -c $(CFLAGS) gui_init.c)
	$(MAKE) update
	$(MAKE) "GUI_FAKES=special/gui_fakes.o" all

debug:	.FRC
	(cd ccr; $(CC) -c -g $(CFLAGS) gui_init.c)
	$(MAKE) update
	$(MAKE) "DEBUG=-g" "FFLAGS = $(FFLAGS) -Ddebug"  \
		"CFLAGS = -g $(CFLAGS)" \
		"GUI_FAKES=special/gui_fakes.o" all

gui_nodebug: .FRC
	(cd ccr; $(CC) -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = $(CFLAGS) -DLINK_GUI_AS_MAIN" \
		"GUILIB = -L../lib -lgui ../list-2.1/liblist.a" \
		"MOVIELIB=" "EXECNAME=ferretgui_c" "GUI_FAKES=" \
		"GUI=GUI" all
gui_debug: .FRC
	(cd ccr; $(CC) -g -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = -g $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG=-g "\
		"GUILIB = -L../lib -lgui ../list-2.1/liblist.a" \
		"MOVIELIB=" "EXECNAME=ferretgui_c" \
		"FFLAGS = $(FFLAGS) -Ddebug" "GUI=GUI" "DEBUG_TITLE=alpha" \
		"GUI_FAKES="  all

DODS_gui_debug: .FRC
	(cd ccr; $(CC) -g -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = -g $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG=-g "\
		"EXECNAME=ferretdods_gui" \
		"GUILIB = -L../lib -lgui ../list-2.1/liblist.a" \
		"MOVIELIB="  \
                "LD = g++" "CDFLIB = $(DODS_LIBS)" \
		"FFLAGS = $(FFLAGS) -Ddebug" "GUI_FAKES=" \
		"DEBUG_TITLE=alpha" all

DODS_debug: .FRC
	(cd ccr; $(CC) -g -c $(CFLAGS) gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = -g $(CFLAGS)" "DEBUG=-g "\
		"EXECNAME=ferretdods"  \
		"GUI_FAKES=special/gui_fakes.o"\
                "LD = g++" "CDFLIB = $(DODS_LIBS)" \
		"FFLAGS = $(FFLAGS) -Ddebug"  \
		"DEBUG_TITLE=alpha" all

DODS_gui_debug_valgrind: .FRC
	(cd ccr; $(CC) -O0 -g -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = -O0 -g $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG=-g -O0"\
		"EXECNAME=ferretdods_gui" \
		"GUILIB = -L../lib -lgui ../list-2.1/liblist.a" \
		"MOVIELIB="  \
                "LD = g++" "CDFLIB = $(DODS_LIBS)" \
		"FFLAGS = $(FFLAGS) -O0  -Ddebug" "GUI_FAKES=" \
		"DEBUG_TITLE=alpha" all
                
profiling: .FRC
	(cd ccr; $(CC) -g $(PROF_FLAGS) -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = -g $(PROF_FLAGS) $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG=-g "\
		"EXECNAME=ferretdods_gui" \
		"GUILIB = -L../lib -lgui ../list-2.1/liblist.a" \
		"MOVIELIB="  \
                "LD = g++ $(PROF_FLAGS) " "CDFLIB = $(DODS_LIBS)" \
		"FFLAGS = $(PROF_FLAGS) $(FFLAGS) -Ddebug" "GUI_FAKES=" \
		"DEBUG_TITLE=alpha" all

DODS_gui: .FRC
	(cd ccr; $(CC) -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG="\
		"EXECNAME=ferretdods_gui" \
		"GUILIB = -L../lib -lgui ../list-2.1/liblist.a" \
		"MOVIELIB="  \
                "LD = g++" "CDFLIB = $(DODS_LIBS)" \
		"FFLAGS = $(FFLAGS)" "GUI_FAKES=" \
		"DEBUG_TITLE=alpha" all

DODS_only: .FRC
	(cd ccr; $(CC) -c $(CFLAGS) gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = $(CFLAGS)" "DEBUG="\
		"EXECNAME=ferretdods"  \
		"GUI_FAKES=special/gui_fakes.o"\
                "LD = g++" "CDFLIB = $(DODS_LIBS)" \
		"FFLAGS = $(FFLAGS) -Ddebug"  \
		"DEBUG_TITLE=alpha" all

DODS_gui_static: .FRC
	(cd ccr; $(CC) -c $(CFLAGS) -DLINK_GUI_AS_MAIN gui_init.c)
	$(MAKE) update
	$(MAKE) "CFLAGS = $(CFLAGS) -DLINK_GUI_AS_MAIN" "DEBUG="\
		"EXECNAME=ferretdods_gui" \
		"GUILIB = -L../lib -lgui ../list-2.1/liblist.a" \
		"MOVIELIB="  \
                "LD = g++" "LDFLAGS=-v --verbose -L/lib -L/usr/X11R6/lib64  -static" \
                "CDFLIB = $(DODS_LIBS)" \
		"FFLAGS = $(FFLAGS)" "GUI_FAKES=" \
		all

fortran: .FRC
	$(MAKE) "FERMAIN=special/fermain_fortran.o" "EXECNAME=ferret_fortran" all


ferret:	
	- mv $(EXECNAME) $(EXECNAME)_pre_`date +'%d%h'|tr '[A-Z]' '[a-z]'`
	- $(LD) $(LDFLAGS) -o $(EXECNAME) \
		$(FERMAIN) \
		$(FEROBJS) \
		$(GUI_FAKES) \
		$(XTRA_OBJ) \
		$(GUILIB) \
		$(GKSOBJ) \
		$(PPLOBJS) \
		$(EFUOBJS) \
		$(EFIOBJS) \
		$(BDEAD) \
		$(FERLIBS) \
		$(PPLLIBS) \
		$(MOVIELIB) \
		$(TMLIB) \
		$(READLINELIB) \
		$(TERMCAPLIB) \
		$(CDFLIB) \
		$(HDFLIB) \
		$(GKSLIB) \
		$(SYSLIB) \
		$(DODS_SYSLIB)
	chmod +x $(EXECNAME)

update: .FRC
	${MAKE} update_date DATE=`/bin/date +%D`
	rm -f special/xrevision_type_data.[Ffo]
	sed -e 's&mmmmmmmm&($(DEBUG_TITLE)/$(GUI))&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	rm -f special/xplatform_type_data.[Ffo]
	sed -e 's&mmmmmmmm&$(PLATFORM)&' -e 's&(/&(&' -e 's&/)&)&' < special/xplatform_type_data.template > special/xplatform_type_data.F 
	(cd special;make)

update_date:
	rm -f special/xmake_date_data.[Ffo]
	sed "s&xxxxxxxx&$(DATE)&" <special/xmake_date_data.template >special/xmake_date_data.F


libraries: ppl_libraries fer_libraries


fmt_libraries: .FRC
	(cd ../fmt/src ; \
	$(MAKE) "CC=$(CC)" "CFLAGS=$(DEBUG) $(CFLAGS)" \
		"FFLAGS=$(DEBUG) $(FFLAGS)");

fer_libraries: .FRC
	for i in $(LIBS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; \
        $(MAKE) "MFLAGS=$(MFLAGS)" "CC=$(CC)" "CFLAGS=$(DEBUG) $(CFLAGS) $(GKS_INCLUDES)" \
                "FFLAGS=$(DEBUG) $(FFLAGS)" "PPLUS_FFLAGS=$(PPLUS_FFLAGS)"); \
        done


ppl_libraries: .FRC
	(cd ../ppl ; $(MAKE) "PPLUS_FFLAGS=$(DEBUG) $(PPLUS_FFLAGS) $(GKS_INCLUDES)" \
                              "CFLAGS=$(DEBUG) $(CFLAGS) $(GKS_INCLUDES)" all);


special_code: .FRC
	(cd special ; $(MAKE) "FFLAGS=$(DEBUG) $(FFLAGS)" all);


clean: fmt_clean ppl_clean fer_clean 
	(cd ../lib ; rm -f *.a *.so);

ppl_clean: .FRC
	(cd ../ppl ; $(MAKE) clean);

fer_clean: .FRC
	for i in $(CLEANDIRS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; rm -f *.f; $(MAKE) clean); \
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
