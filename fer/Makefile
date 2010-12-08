#

#
# Site-specific defines
#
include ../site_specific.mk

#
# Platform-specific defines
#
include ../platform_specific.mk.$(HOSTTYPE)

DATE = $(shell /bin/date +%D)

#
# Macros
#

GUI_FAKES = special/gui_fakes.o

DIRS	= ccr ctx dat doo efi efn ef_utility fmt gnl ino mem  \
	ocn plt rpn special stk utl xeq threddsBrowser

LIBS	= ccr ctx dat doo efi efn ef_utility fmt gnl ino mem \
	ocn plt rpn stk utl xeq

CLEANDIRS = $(DIRS)

DEPDIRS = ccr

FERLIBS = -L../lib -Xlinker --start-group -lxeq -lgnl -lferplt -lrpn -lstk -ldoo -locn \
	-lctx -lfmt -lino -lmem -lutl -ldat -lccr -lefi -lefn -ltmap \
	-lplt -lpll -lsym -lcmp -lour -lepi -lusr -Xlinker --end-group

FERMAIN = ccr/fermain_c.o ccr/gui_init.o

FEROBJS = ../ppl/plot/ppldata.o \
	$(wildcard ../fmt/src/x*_data.o) \
	$(wildcard dat/x*_data.o) \
	$(wildcard special/x*_data.o) \
	special/fakes3.o \
	special/ferret_dispatch.o \
	special/ferret_query_f.o \
	special/linux_routines.o

PPLOBJS =  $(wildcard ../ppl/tmapadds/*.o)

EFUOBJS = $(wildcard ef_utility/*.o)

EFIOBJS = $(wildcard efi/*.o)

EXECNAME = ferret_c

EF_CMNS = common/EF_Util.cmn common/EF_Util.parm common/EF_mem_subsc.cmn


#
# Targets
#


.PHONY : all
all: ferret


.PHONY : build
build: update special_code ppl_libraries fmt_libraries fer_libraries threddsBrowser


.PHONY : optimized
optimized:
	$(MAKE) "CFLAGS += -O2" "FFLAGS += -O2" all


.PHONY : debug
debug:
	$(MAKE) "CFLAGS += -O0 -g" "FFLAGS += -O0 -g" all


.PHONY : optimizedbuild
optimizedbuild:
	$(MAKE) "CFLAGS += -O2" "FFLAGS += -O2" build


.PHONY : debugbuild
debugbuild:
	$(MAKE) "CFLAGS += -O0 -g" "FFLAGS += -O0 -g" build


.PHONY : ferret
ferret: build
	$(CC) $(CFLAGS) -c ccr/gui_init.c -o ccr/gui_init.o
	$(CC) $(CFLAGS) -c ccr/fermain_c.c -o ccr/fermain_c.o
	$(LD) $(LDFLAGS) \
	      $(FERMAIN) \
	      $(FEROBJS) \
	      $(GUI_FAKES) \
	      $(XTRA_OBJ) \
	      $(PPLOBJS) \
	      $(EFUOBJS) \
	      $(EFIOBJS) \
	      $(FERLIBS) \
	      $(GKSLIB) \
	      $(SYSLIB) \
	      -o $(EXECNAME)

.PHONY : update
update:
	rm -f special/xmake_date_data.[Ffo]
	sed "s&xxxxxxxx&$(DATE)&" < special/xmake_date_data.template > special/xmake_date_data.F
	rm -f special/xrevision_type_data.[Ffo]
	sed -e 's&mmmmmmmm&&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	rm -f special/xplatform_type_data.[Ffo]
	sed -e 's&mmmmmmmm&$(PLATFORM)&' -e 's&(/&(&' -e 's&/)&)&' < special/xplatform_type_data.template > special/xplatform_type_data.F 


.PHONY : special_code
special_code: update
	$(MAKE) -C special all


.PHONY : ppl_libraries
ppl_libraries:
	$(MAKE) -C ../ppl all


.PHONY : fmt_libraries
fmt_libraries:
	$(MAKE) -C ../fmt/src


.PHONY : fer_libraries
fer_libraries: update
	for i in $(LIBS) ; do \
            $(MAKE) -C $$i ; \
        done


.PHONY : threddsBrowser
threddsBrowser:
	$(MAKE) -C threddsBrowser all


.PHONY : clean
clean: ppl_clean fmt_clean fer_clean 
	rm -f ../lib/*


.PHONY : ppl_clean
ppl_clean:
	$(MAKE) -C ../ppl clean


.PHONY : fmt_clean
fmt_clean:
	$(MAKE) -C ../fmt/src clean


.PHONY : fer_clean
fer_clean:
	for i in $(CLEANDIRS) ; do \
            $(MAKE) -C $$i clean ; \
        done
	rm -f special/xmake_date_data.F special/xrevision_type_data.F special/xplatform_type_data.F

