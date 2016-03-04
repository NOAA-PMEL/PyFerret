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
#  v6.6 07/10 *kms* - added threddsBrowser to DIRS (for clean)
#                     created threddsBrowser target and added it to all
# v6.7 4/11 *kms* - renamed old "all" target to "build" and made the "all"
#                   target go to the desired default target ncdf4;
#                   cleaned-up tags
#  v6.8  7/13 *kms* - removed threddsBrowser (moved to top level since it is a separate app)
#  v6.97 2/16 *acm* - restore "profiling" target
#  V698  3/16 *acm* Removed Philander-Seigel code including ocn/ directory
# 
# include platform specific macro definitions
#

include ../site_specific.mk
include ../platform_specific.mk.$(BUILDTYPE)

#
# Macros
#

#*kob* 2/99 - set GUI_FAKES so that by default it is assumed no gui
#	      stuff is needed.  in order to allow the "make" command 
#	      by itself to work.

GUI_FAKES = special/gui_fakes.o

DIRS	= ccr ctx dat doo efi efn ef_utility fmt gnl ino mem  \
	 plt rpn special stk utl xeq

LIBS	= ccr ctx dat doo efi efn ef_utility fmt gnl ino mem \
	 plt rpn stk utl xeq

CLEANDIRS = $(DIRS)

DEPDIRS = ccr

FERLIBS = -lxeq -lgnl  -lccr -lferplt -lrpn -lstk -ldoo  \
	-lctx -lfmt -lino -lmem -lutl -ldat -lccr -lefi -lefn

PPLLIBS = -lplt -lpll -lsym -lcmp -lour -lepi -lusr -ltmap

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

EF_CMNS = common/EF_Util.cmn common/EF_Util.parm common/EF_mem_subsc.cmn

#
# Targets
#

.PHONY: all
all: ncdf4

.PHONY: build
build: update fmt_libraries fer_libraries ppl_libraries special_code ferret

.PHONY: ncdf4
ncdf4:
	mkdir -p ../lib
	$(MAKE) "CFLAGS=$(CFLAGS) -O -DNDEBUG" \
		"FFLAGS=$(FFLAGS) -O -DNDEBUG" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) -O" \
		"DEBUG_TITLE = opt" \
		"GUI_FAKES=special/gui_fakes.o" \
		build

.PHONY: beta
beta:
	mkdir -p ../lib
	$(MAKE) "CFLAGS=$(CFLAGS) -O -DNDEBUG" \
		"FFLAGS=$(FFLAGS) -O -DNDEBUG" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) -O" \
		"DEBUG_TITLE = beta/opt" \
		"GUI_FAKES=special/gui_fakes.o" \
		build

.PHONY: debug
debug:
	mkdir -p ../lib
	$(MAKE) "CFLAGS = $(CFLAGS) -O0 -g"  \
		"FFLAGS = $(FFLAGS) -O0 -g -fbounds-check" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) -O0 -g" \
		"DEBUG_TITLE = beta/dbg" \
		"GUI_FAKES=special/gui_fakes.o" \
		build

.PHONY: profiling
profiling:
	mkdir -p ../lib
	$(MAKE) "CFLAGS = $(CFLAGS) -O0 -g $(PROF_FLAGS) "  \
		"FFLAGS = $(FFLAGS) -O0 -g $(PROF_FLAGS) -fbounds-check" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) -O0 -g $(PROF_FLAGS)" \
		"LDFLAGS = $(PPLUS_FFLAGS) $(PROF_FLAGS)" \
		"DEBUG_TITLE = profiling" \
		"GUI_FAKES=special/gui_fakes.o" build

.PHONY: ferret
ferret:	
	- mv $(EXECNAME) $(EXECNAME)_`ls -l $(EXECNAME) | awk '{print $$6"_"$$7}' | tr [A-Z] [a-z]`
	- $(LD) $(LDFLAGS) -L../lib \
		$(FERMAIN) \
		$(FEROBJS) \
		$(GUI_FAKES) \
		$(PPLOBJS) \
		$(EFUOBJS) \
		$(EFIOBJS) \
		$(FERLIBS) \
		$(PPLLIBS) \
		$(READLINELIB) \
		$(TERMCAPLIB) \
		$(CDFLIB) \
		$(HDFLIB) \
		$(GKSLIB) \
		$(SYSLIB) \
		-o $(EXECNAME)
	chmod +x $(EXECNAME)

.PHONY: update
update:
	- mv $(EXECNAME) $(EXECNAME)_`ls -l $(EXECNAME) | awk '{print $$6"_"$$7}' | tr [A-Z] [a-z]`
	${MAKE} update_date DATE=`/bin/date +%D`
	rm -f special/xrevision_type_data.[Ffo]
	sed -e 's&mmmmmmmm&($(DEBUG_TITLE)/$(GUI))&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	rm -f special/xplatform_type_data.[Ffo]
	sed -e 's&mmmmmmmm&$(PLATFORM)&' -e 's&(/&(&' -e 's&/)&)&' < special/xplatform_type_data.template > special/xplatform_type_data.F 
	(cd special; make)

.PHONY: update_date
update_date:
	rm -f special/xmake_date_data.[Ffo]
	sed "s&xxxxxxxx&$(DATE)&" <special/xmake_date_data.template >special/xmake_date_data.F

.PHONY: libraries
libraries: ppl_libraries fer_libraries

.PHONY: fmt_libraries
fmt_libraries:
	$(MAKE) -C ../fmt/src

.PHONY: fer_libraries
fer_libraries:
	for i in $(LIBS) ; \
        do \
            if ! $(MAKE) -C $$i ; then \
                exit 1 ; \
            fi ; \
        done

.PHONY: ppl_libraries
ppl_libraries:
	$(MAKE) -C ../ppl all

.PHONY: special_code
special_code:
	$(MAKE) -C special all

.PHONY: clean
clean: fmt_clean ppl_clean fer_clean 
	- mv $(EXECNAME) $(EXECNAME)_`ls -l $(EXECNAME) | awk '{print $$6"_"$$7}' | tr [A-Z] [a-z]`
	rm -f ../lib/*.a ../lib/*.so
	rm -f special/xmake_date_data.F special/xplatform_type_data.F special/xrevision_type_data.F

.PHONY: ppl_clean
ppl_clean:
	$(MAKE) -C ../ppl clean

.PHONY: fer_clean
fer_clean:
	for i in $(CLEANDIRS) ; \
        do \
            $(MAKE) -C $$i clean ; \
        done

.PHONY: fmt_clean
fmt_clean:
	$(MAKE) -C ../fmt/src clean

#
# End of Makefile
#
