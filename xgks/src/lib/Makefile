# Generated automatically from Makefile.in by configure.
# Makefile for the C, XGKS library
#
# $Id$

LIBRARY		= xgks

FONTDBDIR	= "$(LIBDIR)/xgksfonts"
CPPFLAGS	= -I../../port -DFONTDBDIR='$(FONTDBDIR)' -I/usr/include -DNDEBUG
CFLAGS		= -g -O

HEADERS		= xgks.h
headers		= event.h fillarea.h font.h gks_defines.h gks_errors.h \
		  gks_implem.h gkslist.h input.h metafile.h \
		  polylines.h polymarkers.h primitive.h text.h wdt.h \
		  wslist.h
csrcs		= act_ws.c aspect_flags.c cellarray.c choice.c colours.c \
		  deferral_ws.c escape.c event.c externinit.c fillarea.c \
		  gdp.c ggdp.c gks_error.c input.c inqWDT.c \
		  inqfillareas.c inqpixel.c inqpmarker.c inqpolylines.c \
		  inqtext.c inqtransform.c inquiries.c locator.c \
		  message.c metafile.c mkerrmsg.c open_gks.c open_ws.c pick.c \
		  polylines.c polymarkers.c prmgr.c segments.c string.c \
		  stroke.c text.c transforms.c umalloc.c update.c \
		  valuator.c
LIBOBJS		= act_ws.o aspect_flags.o cellarray.o choice.o colours.o \
		  deferral_ws.o escape.o event.o externinit.o fillarea.o \
		  gdp.o ggdp.o gks_error.o input.o inqWDT.o \
		  inqfillareas.o inqpixel.o inqpmarker.o inqpolylines.o \
		  inqtext.o inqtransform.o inquiries.o locator.o \
		  message.o metafile.o open_gks.o open_ws.o pick.o \
		  polylines.o polymarkers.o prmgr.o segments.o string.o \
		  stroke.o text.o transforms.o umalloc.o update.o \
		  valuator.o
SUBDIRS		= cgm gksm x
SUBDIR_TARGETS	= cgm/all cgm/clean cgm/distclean \
		  gksm/all gksm/clean gksm/distclean \
		  x/all x/clean x/distclean

MANIFEST	= Makefile.in depend $(HEADERS) $(headers) $(csrcs)

all::		gkserrmsg.h cgm/all gksm/all x/all library 

install::	installed_library installed_headers

clean::		cgm/clean gksm/clean x/clean
	rm -f gkserrmsg.h mkerrmsg

distclean::	cgm/distclean gksm/clean x/distclean

gkserrmsg.h:	gks_errors.h mkerrmsg
	./mkerrmsg < gks_errors.h > $@

mkerrmsg:	mkerrmsg.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ mkerrmsg.c

# Generated automatically from master.mk.in by configure.
# This is the master makefile.  It contains general rules.
#
# $Id$


.SUFFIXES:
.SUFFIXES:	.so .o .i .c .fc .f .F .l .y

############################################################################
# General Macros:
############################################################################
prefix		= /home/lotus/kobrien/xgks/lib
exec_prefix	= /home/lotus/kobrien/xgks/lib/bin
AR		= ar
CC		= cc
CPP		= cc -E
INCDIR		= $(prefix)/include
INSTALL		= cp
INSTALL_DATA	= $(INSTALL)
INSTALL_PROGRAM	= $(INSTALL)
FC		= f77
NCDUMP		= @NCDUMP@
NCGEN		= @NCGEN@
NEQN		= neqn
OS		= osf
SHELL		= /bin/sh
TBL		= tbl


############################################################################
# Recursive make(1)s in Subdirectories:
############################################################################

subdir_dummy_target $(SUBDIR_TARGETS):
	@echo ""
	@dir=`echo $@ | sed 's,/.*,,'`; \
	    target=`echo $@ | sed "s,$$dir/,,"`; \
	    cd $$dir && \
	    echo "making \`$$target' in directory `pwd`" && \
	    echo "" && \
	    $(MAKE) $(MFLAGS) prefix=$(prefix) exec_prefix=$(exec_prefix) \
		$$target || exit 1
	@echo ""
	@echo "returning to directory `pwd`"
	@echo ""


############################################################################
# Compilation:
############################################################################
FORTC		= /home/lotus/kobrien/xgks/osf/port/fortc/fortc

# Compile FORTRAN code.
#
.f.o:
	$(FC) -c $(FFLAGS) $<
.F.o:
	ln -s $< $$$$.c \
	&& $(CPP) -DUD_FORTRAN_SOURCE $(CPPFLAGS) $$$$.c \
	    | sed '/^# *[0-9]/d' > $*.f \
	|| (rm -f $*.f $$$$.c; exit 1) \
	&& rm $$$$.c
	$(FC) -c $(FFLAGS) $*.f
.F.f:
	ln -s $< $$$$.c \
	&& $(CPP) -DUD_FORTRAN_SOURCE $(CPPFLAGS) $$$$.c \
	    | sed '/^# *[0-9]/d' > $@ \
	|| (rm -f $@ $$$$.c; exit 1) \
	&& rm $$$$.c

# Generate FORTRAN-callable C code.
#
.fc.o:
	$(FORTC) -O $(OS) $< > $*.c \
	    || ( rm -f $*.c; exit 1)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $*.c
.fc.c:
	$(FORTC) -O $(OS) $< > $@ \
	    || (rm -f $*.c; exit 1)

# Compile C code.
#
.c.o:
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $<

# Compile lex(1) code.  This rule is here to ensure the use of CPPFLAGS.
# 
.l.o:
	rm -f $*.c
	$(LEX) -t $< > $*.c
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $*.c
	rm $*.c

# C preprocess.  
#   NB: 
#	1. The output `file' is not created but appears on standard output;
#	2. Line numbering information is kept.
#
.c.i:
	$(CPP) $(CPPFLAGS) $<


############################################################################
# Libraries:
############################################################################
RANLIB		= ranlib
LIBDIR		= $(prefix)/lib
LIB_A		= lib$(LIBRARY).a

library::		$(LIB_A)

# The following rule is forced because the library can be created by a
# makefile in another directory prior to the execution of the following rule.
#
$(LIB_A):		$(LIBOBJS) FORCE
	$(AR) rcuv $@ $(LIBOBJS)
	$(RANLIB) $@

# The following rule is for those makefiles needing to install object
# modules in a library in another directory.  They need only define 
# REMOTE_LIBRARY and LIBOBJS.  The rule is forced for the same reason as the
# $(LIB_A) rule.
#
lib_dummy $(REMOTE_LIBRARY):	$(LIBOBJS) FORCE
	$(AR) rcuv $@ $(LIBOBJS)
	$(RANLIB) $@

installed_libfiles:	FORCE
	@if [ -n "$(LIBFILES)" ]; then \
	    libfiles="$(LIBFILES)"; \
	    for libfile in $$libfiles; do \
		$(MAKE) $(MFLAGS) LIBDIR=$(LIBDIR) LIBFILE=$$libfile \
		    installed_libfile || exit 1; \
	    done; \
	else \
	    :; \
	fi

installed_library::	FORCE
	@$(MAKE) $(MFLAGS) LIBDIR=$(LIBDIR) LIBFILE=$(LIB_A) \
	    installed_libfile

# The following target decouples the previous targets from the actual rule;
# thus, eliminating extraneous make(1) messages.
#
installed_libfile:	$(LIBDIR)/$(LIBFILE)

$(LIBDIR)/$(LIBFILE):	$(LIBDIR) $(LIBFILE)
	$(INSTALL_DATA) $(LIBFILE) $@

dummy_libdir $(LIBDIR):
	mkdir -p $@
	chmod u+rwx,g+rws,o=rx $@

LIB_SO		= lib$(LIBRARY).so.$(MAJOR_NO).$(MINOR_NO)
LIB_SA		= lib$(LIBRARY).sa.$(MAJOR_NO).$(MINOR_NO)

#library::		FORCE
#	@so_objs=`echo $(LIBOBJS) | sed 's/\.o/.so/g'`; \
#	if [ -z "$$so_objs" ]; then \
#	    :; \
#	else \
#	    $(MAKE) $(MFLAGS) $(LIB_SO) \
#		SO_OBJS="$$so_objs"; \
#	fi
#	@if [ -z "$(SA_OBJS)" ]; then \
#	    :; \
#	else \
#	    $(MAKE) $(MFLAGS) $(LIB_SA); \
#	fi

$(LIB_SO):	$(SO_OBJS)
	ld -assert pure-text $(SO_OBJS) -o $@
$(LIB_SA):	$(SA_OBJS) FORCE
	$(AR) rcuv $@ $(SA_OBJS)

.fc.so:
	$(FORTC) -O $(OS) $< > $*.c \
	    || (rm -f $*.c; exit 1)
	$(CC) -c -pic $(CFLAGS) $(CPPFLAGS) $*.c -o $$$$.o \
	    || (rm -f $$$$.o; exit 1)
	mv $$$$.o $@
.c.so:
	$(CC) -c -pic $(CFLAGS) $(CPPFLAGS) $< -o $$$$.o \
	    || (rm -f $$$$.o; exit 1)
	mv $$$$.o $@

#installed_library::	$(LIBDIR)/$(LIB_SO)
#	@if [ -z "$(SA_OBJS)" ]; then \
#	    :; \
#	else \
#	    $(MAKE) $(MFLAGS) $(LIBDIR)/$(LIB_SA); \
#	fi

$(LIBDIR)/$(LIB_SO):	$(LIBDIR) $(LIB_SO)
	$(INSTALL_DATA) $(LIB_SO) $(LIBDIR)/$(LIB_SO)
$(LIBDIR)/$(LIB_SA):	$(LIBDIR) $(LIB_SA)
	$(INSTALL_DATA) $(LIB_SA) $(LIBDIR)/$(LIB_SA)


############################################################################
# Programs:
############################################################################
BINDIR		= $(exec_prefix)

programs:	$(PROGRAMS)
program:
	@if [ -z "$(PROGRAM)" ]; then \
	    : true; \
	else \
	    if [ -z "$(LIBRARY)" ]; then \
		: true; \
	    else \
		$(MAKE) $(MFLAGS) library; \
	    fi; \
	    $(MAKE) $(MFLAGS) $(PROGRAM); \
	fi

dummy_prog $(PROGRAM):	$(OBJS)
	@if [ -r $@.f -o -r $@.F ]; then \
	    ld='$(FC) $(FFLAGS)'; \
	else \
	    ld='$(CC) $(CFLAGS)'; \
	fi; \
	cmd="$$ld $(OBJS) $(LIBS) -o $@"; \
	echo $$cmd; \
	$$cmd \
	    || (rm -f $@; exit 1)

installed_programs:	FORCE
	@$(MAKE) $(MFLAGS) installed_binfiles \
	    BINDIR=$(BINDIR) BINFILES="$(PROGRAMS)"

installed_binfiles:	FORCE
	@if [ -n "$(BINFILES)" ]; then \
	    binfiles="$(BINFILES)"; \
	    for binfile in $$binfiles; do \
		$(MAKE) $(MFLAGS) BINDIR=$(BINDIR) BINFILE=$$binfile \
		    installed_binfile || exit 1; \
	    done; \
	else \
	    :; \
	fi

installed_program:	FORCE
	@$(MAKE) $(MFLAGS) BINDIR=$(BINDIR) BINFILE=$(PROGRAM) \
	    installed_binfile

# The following target decouples the previous targets from the actual rule;
# thus, eliminating extraneous make(1) messages.
#
installed_binfile:	$(BINDIR)/$(BINFILE)

$(BINDIR)/$(BINFILE):	$(BINDIR) $(BINFILE)
	$(INSTALL_PROGRAM) $(BINFILE) $@

dummy_bindir $(BINDIR):
	mkdir -p $@
	chmod u+rwx,g+rws,o=rx $@


############################################################################
# Header files:
############################################################################
installed_headers:	FORCE
	@if [ -n "$(HEADERS)" ]; then \
	    headers="$(HEADERS)"; \
	    for header in $$headers; do \
		$(MAKE) $(MFLAGS) INCDIR=$(INCDIR) HEADER=$$header \
		    installed_header || exit 1; \
	    done; \
	else \
	    :; \
	fi

# The following target decouples the previous targets from the actual rule;
# thus, eliminating extraneous make(1) messages.
#
installed_header:	$(INCDIR)/$(HEADER)

$(INCDIR)/$(HEADER):	$(INCDIR) $(HEADER)
	$(INSTALL_DATA) $(HEADER) $@

dummy_incdir $(INCDIR):
	mkdir -p $@
	chmod u+rwx,g+rws,o=rx $@


############################################################################
# `etc/' files:
############################################################################
ETCDIR	= $(prefix)/etc

installed_etcfiles:
	@if [ -n "$(ETCFILES)" ]; then \
	    etcfiles="$(ETCFILES)"; \
	    for etcfile in $$etcfiles; do \
		$(MAKE) $(MFLAGS) ETCDIR=$(ETCDIR) ETCFILE=$$etcfile \
		    installed_etcfile || exit 1; \
	    done; \
	else \
	    :; \
	fi

# The following target decouples the previous targets from the actual rule;
# thus, eliminating extraneous make(1) messages.
#
installed_etcfile:	$(ETCDIR)/$(ETCFILE)

$(ETCDIR)/$(ETCFILE):	$(ETCDIR) $(ETCFILE)
	$(INSTALL_DATA) $(ETCFILE) $@

dummy_etcdir $(ETCDIR):
	mkdir -p $@
	chmod u+rwx,g+rws,o=rx $@


############################################################################
# Manual Pages:
############################################################################
MANDIR	= $(prefix)/man

installed_manuals:	FORCE
	@if [ -n "$(MANUALS)" ]; then \
	    manuals="$(MANUALS)"; \
	    for manual in $$manuals; do \
		mandir=$(MANDIR)/man`echo $$manual | sed 's/.*\.\(.\).*/\1/'`; \
		$(MAKE) $(MFLAGS) MANDIR=$$mandir MANUAL=$$manual \
		    installed_manual || exit 1; \
	    done; \
	else \
	    :; \
	fi

# The following target decouples the previous targets from the actual rule;
# thus, eliminating extraneous make(1) messages.
#
installed_manual:	$(MANDIR)/$(MANUAL)

# The first sed(1) editing command in the following replaces the expanded
# RCS keyword `$Date ...' in the `.TH' macro line with something more 
# appropriate to a manual page.
#
$(MANDIR)/$(MANUAL):	$(MANDIR) $(MANUAL)
	sed \
	 -e '/^\.TH/s/$$D\ate: \([0-9]*\)\/\([0-9]*\)\/\([0-9]*\).*\$$/\1-\2-\3/'\
	 $(MANUAL) | $(TBL) | $(NEQN) > $@

dummy_mandir $(MANDIR):
	mkdir -p $@
	chmod u+rwx,g+rws,o=rx $@


############################################################################
# Cleanup:
############################################################################
clean::
	rm -f $(GARBAGE) $(PROGRAM) a.out core *.o *.a *.so.* *.sa.* *.so \
	    *.log

realclean::	distclean
distclean::
	rm -f $(GARBAGE) $(PROGRAM) a.out core *.o *.a *.so.* *.sa.* *.so \
	    *.log Makefile config.status cc_id tags


############################################################################
# Configuration:
############################################################################

# The following target is touched because config.status won't update it if
# it hasn't changed.
#
# The following is commented-out because the HPUX make(1) goes into a loop.
#
#Makefile:	config.status
#	./config.status
#	touch $@

config.status:	cc_id configure
	./configure --no-create

cc_id:		FORCE
	if test -r $@ && test "`cat $@`" = "$(CC)"; then \
	    :; \
	else \
	    echo "$(CC)" > $@; \
	fi

configure:		configure.in
	autoconf


############################################################################
# Dependency maintenance:
############################################################################
.PRECIOUS:	depend
#
# Note: The following rule will only work if the C compiler behaves correctly
# when given the `-M' option.  AIX's cc(1) and SunOS's acc(1) behave
# incorrectly.  --Steve Emmerson 1992-09-24
#
# Furthermore, we do not attempt to maintain dependencies on installed
# header files (e.g. those in $(INCDIR)) for two reasons: 1) the dependency
# list generated at Unidata mightn't correspond to the situation at a client
# site; and 2) clients using broken compilers (c.f. previous note) can't
# regenerate the dependency list.
#
depend:		FORCE
	-incdir=`echo $(INCDIR) | sed s',/,\\\/,g'`; \
	cc -M $(CPPFLAGS) *.c 2> /dev/null | \
	    awk '{ \
		if ($$0 ~ /:/) { \
		    target	= $$1; \
		    if (target ~ /:/) { \
			target	= substr(target,1,length(target)-1); \
			start	= 2; \
		    } else { \
			start	= 3; \
		    } \
		} else { \
		    start	= 1; \
		} \
		for (i = start; i <= NF; ++i) { \
		    if ($$i !~ /^\\/ && \
			$$i !~ /port\// && \
			$$i !~ /^\//) \
			print target ": " $$i \
		} \
	    }' > $@
	tag='### DO NOT DELETE THIS LINE.  make depend DEPENDS ON IT ###'; \
	    sed -e "/^$$tag$$/r depend" \
		-e "/^$$tag$$/q" Makefile > Makefile.new
	mv Makefile.new Makefile


############################################################################
# Distributions:
############################################################################
FTPDIR		= /home/ftp/pub
VERSION		= 2.5.5

ftp:		$(FTPDIR)/$(PACKAGE)-$(VERSION).tar.Z
	test ! -h $(FTPDIR)/$(PACKAGE).tar.Z && exit 0; \
	cd $(FTPDIR) || exit 1; \
	rm $(PACKAGE).tar.Z || exit 1; \
	ln -s $(PACKAGE)-$(VERSION).tar.Z $(PACKAGE).tar.Z;

$(FTPDIR)/$(PACKAGE)-$(VERSION).tar.Z:	$(PACKAGE)-$(VERSION).tar.Z
	cp $(PACKAGE)-$(VERSION).tar.Z $@ 
	chmod u+rw,g+rw,o=r $@

tar.Z:		$(PACKAGE)-$(VERSION).tar.Z

$(PACKAGE)-$(VERSION).tar.Z:	MANIFEST
	dir=`basename \`pwd\`` \
	&& id=$(PACKAGE)-$(VERSION) \
	&& cd .. \
	&& ln -s $$dir $$id \
	&& tar -chof - `sed "s|^|$$id/|" $$id/MANIFEST` \
	    | compress > $$id/$@ \
	&& rm $$id

MANIFEST:	FORCE
	@echo 1>&2 Creating MANIFEST
	@$(MAKE) -s $(MFLAGS) MANIFEST.echo > $@

MANIFEST.echo:	FORCE
	@$(MAKE) -s $(MFLAGS) \
	    MANIFEST=`echo $(MANIFEST) | sed 's/MANIFEST//'` \
	    ensure_manifest > /dev/null
	@echo $(MANIFEST) | fmt -1
	@if [ -n "$(SUBDIRS)" ]; then \
	    subdirs="$(SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && \
		$(MAKE) $(MFLAGS) MANIFEST.echo | \
		    sed "s|^|$$subdir/|") || exit 1; \
	    done; \
	else \
	    :; \
	fi

ensure_manifest:	$(MANIFEST)


############################################################################
# Miscellaneous:
############################################################################

# The following dummy target is useful as a dependency to ensure that a
# rule is always executed.
#
FORCE:

### Everything after the following line might be overwritten ###
### DO NOT DELETE THIS LINE.  make depend DEPENDS ON IT ###
act_ws.o: act_ws.c
act_ws.o: ./gks_implem.h
act_ws.o: ./xgks.h
act_ws.o: ./gks_defines.h
act_ws.o: ./primitive.h
act_ws.o: ./input.h
act_ws.o: ./wdt.h
act_ws.o: ./wslist.h
act_ws.o: ./gkslist.h
act_ws.o: ./gks_errors.h
act_ws.o: ./event.h
act_ws.o: ./metafile.h
aspect_flags.o: aspect_flags.c
aspect_flags.o: ./gks_implem.h
aspect_flags.o: ./xgks.h
aspect_flags.o: ./gks_defines.h
aspect_flags.o: ./primitive.h
aspect_flags.o: ./input.h
aspect_flags.o: ./wdt.h
aspect_flags.o: ./wslist.h
aspect_flags.o: ./gkslist.h
aspect_flags.o: ./gks_errors.h
aspect_flags.o: ./event.h
aspect_flags.o: ./metafile.h
cellarray.o: cellarray.c
cellarray.o: ./gks_implem.h
cellarray.o: ./xgks.h
cellarray.o: ./gks_defines.h
cellarray.o: ./primitive.h
cellarray.o: ./input.h
cellarray.o: ./wdt.h
cellarray.o: ./wslist.h
cellarray.o: ./gkslist.h
cellarray.o: ./gks_errors.h
cellarray.o: ./event.h
cellarray.o: ./metafile.h
choice.o: choice.c
choice.o: ./gks_implem.h
choice.o: ./xgks.h
choice.o: ./gks_defines.h
choice.o: ./primitive.h
choice.o: ./input.h
choice.o: ./wdt.h
choice.o: ./wslist.h
choice.o: ./gkslist.h
choice.o: ./gks_errors.h
choice.o: ./event.h
choice.o: ./metafile.h
colours.o: colours.c
colours.o: ./gks_implem.h
colours.o: ./xgks.h
colours.o: ./gks_defines.h
colours.o: ./primitive.h
colours.o: ./input.h
colours.o: ./wdt.h
colours.o: ./wslist.h
colours.o: ./gkslist.h
colours.o: ./gks_errors.h
colours.o: ./event.h
colours.o: ./metafile.h
deferral_ws.o: deferral_ws.c
deferral_ws.o: ./gks_implem.h
deferral_ws.o: ./xgks.h
deferral_ws.o: ./gks_defines.h
deferral_ws.o: ./primitive.h
deferral_ws.o: ./input.h
deferral_ws.o: ./wdt.h
deferral_ws.o: ./wslist.h
deferral_ws.o: ./gkslist.h
deferral_ws.o: ./gks_errors.h
deferral_ws.o: ./event.h
deferral_ws.o: ./metafile.h
escape.o: escape.c
escape.o: ./gks_implem.h
escape.o: ./xgks.h
escape.o: ./gks_defines.h
escape.o: ./primitive.h
escape.o: ./input.h
escape.o: ./wdt.h
escape.o: ./wslist.h
escape.o: ./gkslist.h
escape.o: ./gks_errors.h
escape.o: ./event.h
escape.o: ./metafile.h
event.o: event.c
event.o: ./gks_implem.h
event.o: ./xgks.h
event.o: ./gks_defines.h
event.o: ./primitive.h
event.o: ./input.h
event.o: ./wdt.h
event.o: ./wslist.h
event.o: ./gkslist.h
event.o: ./gks_errors.h
event.o: ./event.h
event.o: ./metafile.h
externinit.o: externinit.c
externinit.o: ./gks_implem.h
externinit.o: ./xgks.h
externinit.o: ./gks_defines.h
externinit.o: ./primitive.h
externinit.o: ./input.h
externinit.o: ./wdt.h
externinit.o: ./wslist.h
externinit.o: ./gkslist.h
externinit.o: ./gks_errors.h
externinit.o: ./event.h
externinit.o: ./metafile.h
fillarea.o: fillarea.c
fillarea.o: ./gks_implem.h
fillarea.o: ./xgks.h
fillarea.o: ./gks_defines.h
fillarea.o: ./primitive.h
fillarea.o: ./input.h
fillarea.o: ./wdt.h
fillarea.o: ./wslist.h
fillarea.o: ./gkslist.h
fillarea.o: ./gks_errors.h
fillarea.o: ./event.h
fillarea.o: ./metafile.h
fillarea.o: ./fillarea.h
gdp.o: gdp.c
gdp.o: ./gks_implem.h
gdp.o: ./xgks.h
gdp.o: ./gks_defines.h
gdp.o: ./primitive.h
gdp.o: ./input.h
gdp.o: ./wdt.h
gdp.o: ./wslist.h
gdp.o: ./gkslist.h
gdp.o: ./gks_errors.h
gdp.o: ./event.h
gdp.o: ./metafile.h
gerrorhand.o: gerrorhand.c
gerrorhand.o: ./gks_implem.h
gerrorhand.o: ./xgks.h
gerrorhand.o: ./gks_defines.h
gerrorhand.o: ./primitive.h
gerrorhand.o: ./input.h
gerrorhand.o: ./wdt.h
gerrorhand.o: ./wslist.h
gerrorhand.o: ./gkslist.h
gerrorhand.o: ./gks_errors.h
gerrorhand.o: ./event.h
gerrorhand.o: ./metafile.h
ggdp.o: ggdp.c
ggdp.o: ./gks_implem.h
ggdp.o: ./xgks.h
ggdp.o: ./gks_defines.h
ggdp.o: ./primitive.h
ggdp.o: ./input.h
ggdp.o: ./wdt.h
ggdp.o: ./wslist.h
ggdp.o: ./gkslist.h
ggdp.o: ./gks_errors.h
ggdp.o: ./event.h
ggdp.o: ./metafile.h
gks_error.o: gks_error.c
gks_error.o: ./gks_implem.h
gks_error.o: ./xgks.h
gks_error.o: ./gks_defines.h
gks_error.o: ./primitive.h
gks_error.o: ./input.h
gks_error.o: ./wdt.h
gks_error.o: ./wslist.h
gks_error.o: ./gkslist.h
gks_error.o: ./gks_errors.h
gks_error.o: ./event.h
gks_error.o: ./metafile.h
gks_error.o: ./gkserrmsg.h
input.o: input.c
input.o: ./gks_implem.h
input.o: ./xgks.h
input.o: ./gks_defines.h
input.o: ./primitive.h
input.o: ./input.h
input.o: ./wdt.h
input.o: ./wslist.h
input.o: ./gkslist.h
input.o: ./gks_errors.h
input.o: ./event.h
input.o: ./metafile.h
inqWDT.o: inqWDT.c
inqWDT.o: ./gks_implem.h
inqWDT.o: ./xgks.h
inqWDT.o: ./gks_defines.h
inqWDT.o: ./primitive.h
inqWDT.o: ./input.h
inqWDT.o: ./wdt.h
inqWDT.o: ./wslist.h
inqWDT.o: ./gkslist.h
inqWDT.o: ./gks_errors.h
inqWDT.o: ./event.h
inqWDT.o: ./metafile.h
inqfillareas.o: inqfillareas.c
inqfillareas.o: ./gks_implem.h
inqfillareas.o: ./xgks.h
inqfillareas.o: ./gks_defines.h
inqfillareas.o: ./primitive.h
inqfillareas.o: ./input.h
inqfillareas.o: ./wdt.h
inqfillareas.o: ./wslist.h
inqfillareas.o: ./gkslist.h
inqfillareas.o: ./gks_errors.h
inqfillareas.o: ./event.h
inqfillareas.o: ./metafile.h
inqfillareas.o: ./fillarea.h
inqpixel.o: inqpixel.c
inqpixel.o: ./gks_implem.h
inqpixel.o: ./xgks.h
inqpixel.o: ./gks_defines.h
inqpixel.o: ./primitive.h
inqpixel.o: ./input.h
inqpixel.o: ./wdt.h
inqpixel.o: ./wslist.h
inqpixel.o: ./gkslist.h
inqpixel.o: ./gks_errors.h
inqpixel.o: ./event.h
inqpixel.o: ./metafile.h
inqpmarker.o: inqpmarker.c
inqpmarker.o: ./gks_implem.h
inqpmarker.o: ./xgks.h
inqpmarker.o: ./gks_defines.h
inqpmarker.o: ./primitive.h
inqpmarker.o: ./input.h
inqpmarker.o: ./wdt.h
inqpmarker.o: ./wslist.h
inqpmarker.o: ./gkslist.h
inqpmarker.o: ./gks_errors.h
inqpmarker.o: ./event.h
inqpmarker.o: ./metafile.h
inqpmarker.o: ./polymarkers.h
inqpolylines.o: inqpolylines.c
inqpolylines.o: ./gks_implem.h
inqpolylines.o: ./xgks.h
inqpolylines.o: ./gks_defines.h
inqpolylines.o: ./primitive.h
inqpolylines.o: ./input.h
inqpolylines.o: ./wdt.h
inqpolylines.o: ./wslist.h
inqpolylines.o: ./gkslist.h
inqpolylines.o: ./gks_errors.h
inqpolylines.o: ./event.h
inqpolylines.o: ./metafile.h
inqpolylines.o: ./polylines.h
inqtext.o: inqtext.c
inqtext.o: ./gks_implem.h
inqtext.o: ./xgks.h
inqtext.o: ./gks_defines.h
inqtext.o: ./primitive.h
inqtext.o: ./input.h
inqtext.o: ./wdt.h
inqtext.o: ./wslist.h
inqtext.o: ./gkslist.h
inqtext.o: ./gks_errors.h
inqtext.o: ./event.h
inqtext.o: ./metafile.h
inqtext.o: ./text.h
inqtransform.o: inqtransform.c
inqtransform.o: ./gks_implem.h
inqtransform.o: ./xgks.h
inqtransform.o: ./gks_defines.h
inqtransform.o: ./primitive.h
inqtransform.o: ./input.h
inqtransform.o: ./wdt.h
inqtransform.o: ./wslist.h
inqtransform.o: ./gkslist.h
inqtransform.o: ./gks_errors.h
inqtransform.o: ./event.h
inqtransform.o: ./metafile.h
inquiries.o: inquiries.c
inquiries.o: ./gks_implem.h
inquiries.o: ./xgks.h
inquiries.o: ./gks_defines.h
inquiries.o: ./primitive.h
inquiries.o: ./input.h
inquiries.o: ./wdt.h
inquiries.o: ./wslist.h
inquiries.o: ./gkslist.h
inquiries.o: ./gks_errors.h
inquiries.o: ./event.h
inquiries.o: ./metafile.h
locator.o: locator.c
locator.o: ./gks_implem.h
locator.o: ./xgks.h
locator.o: ./gks_defines.h
locator.o: ./primitive.h
locator.o: ./input.h
locator.o: ./wdt.h
locator.o: ./wslist.h
locator.o: ./gkslist.h
locator.o: ./gks_errors.h
locator.o: ./event.h
locator.o: ./metafile.h
message.o: message.c
message.o: ./gks_implem.h
message.o: ./xgks.h
message.o: ./gks_defines.h
message.o: ./primitive.h
message.o: ./input.h
message.o: ./wdt.h
message.o: ./wslist.h
message.o: ./gkslist.h
message.o: ./gks_errors.h
message.o: ./event.h
message.o: ./metafile.h
metafile.o: metafile.c
metafile.o: ./gks_implem.h
metafile.o: ./xgks.h
metafile.o: ./gks_defines.h
metafile.o: ./primitive.h
metafile.o: ./input.h
metafile.o: ./wdt.h
metafile.o: ./wslist.h
metafile.o: ./gkslist.h
metafile.o: ./gks_errors.h
metafile.o: ./event.h
metafile.o: ./metafile.h
metafile.o: ./gksm/gksm.h
metafile.o: ./cgm/cgm.h
mkerrmsg.o: mkerrmsg.c
open_gks.o: open_gks.c
open_gks.o: ./gks_implem.h
open_gks.o: ./xgks.h
open_gks.o: ./gks_defines.h
open_gks.o: ./primitive.h
open_gks.o: ./input.h
open_gks.o: ./wdt.h
open_gks.o: ./wslist.h
open_gks.o: ./gkslist.h
open_gks.o: ./gks_errors.h
open_gks.o: ./event.h
open_gks.o: ./metafile.h
open_gks.o: ./fillarea.h
open_gks.o: ./text.h
open_gks.o: ./polylines.h
open_gks.o: ./polymarkers.h
open_ws.o: open_ws.c
open_ws.o: ./gks_implem.h
open_ws.o: ./xgks.h
open_ws.o: ./gks_defines.h
open_ws.o: ./primitive.h
open_ws.o: ./input.h
open_ws.o: ./wdt.h
open_ws.o: ./wslist.h
open_ws.o: ./gkslist.h
open_ws.o: ./gks_errors.h
open_ws.o: ./event.h
open_ws.o: ./metafile.h
pick.o: pick.c
pick.o: ./gks_implem.h
pick.o: ./xgks.h
pick.o: ./gks_defines.h
pick.o: ./primitive.h
pick.o: ./input.h
pick.o: ./wdt.h
pick.o: ./wslist.h
pick.o: ./gkslist.h
pick.o: ./gks_errors.h
pick.o: ./event.h
pick.o: ./metafile.h
polylines.o: polylines.c
polylines.o: ./gks_implem.h
polylines.o: ./xgks.h
polylines.o: ./gks_defines.h
polylines.o: ./primitive.h
polylines.o: ./input.h
polylines.o: ./wdt.h
polylines.o: ./wslist.h
polylines.o: ./gkslist.h
polylines.o: ./gks_errors.h
polylines.o: ./event.h
polylines.o: ./metafile.h
polylines.o: ./polylines.h
polymarkers.o: polymarkers.c
polymarkers.o: ./gks_implem.h
polymarkers.o: ./xgks.h
polymarkers.o: ./gks_defines.h
polymarkers.o: ./primitive.h
polymarkers.o: ./input.h
polymarkers.o: ./wdt.h
polymarkers.o: ./wslist.h
polymarkers.o: ./gkslist.h
polymarkers.o: ./gks_errors.h
polymarkers.o: ./event.h
polymarkers.o: ./metafile.h
polymarkers.o: ./polymarkers.h
prmgr.o: prmgr.c
prmgr.o: ./gks_implem.h
prmgr.o: ./xgks.h
prmgr.o: ./gks_defines.h
prmgr.o: ./primitive.h
prmgr.o: ./input.h
prmgr.o: ./wdt.h
prmgr.o: ./wslist.h
prmgr.o: ./gkslist.h
prmgr.o: ./gks_errors.h
prmgr.o: ./event.h
prmgr.o: ./metafile.h
segments.o: segments.c
segments.o: ./gks_implem.h
segments.o: ./xgks.h
segments.o: ./gks_defines.h
segments.o: ./primitive.h
segments.o: ./input.h
segments.o: ./wdt.h
segments.o: ./wslist.h
segments.o: ./gkslist.h
segments.o: ./gks_errors.h
segments.o: ./event.h
segments.o: ./metafile.h
string.o: string.c
string.o: ./gks_implem.h
string.o: ./xgks.h
string.o: ./gks_defines.h
string.o: ./primitive.h
string.o: ./input.h
string.o: ./wdt.h
string.o: ./wslist.h
string.o: ./gkslist.h
string.o: ./gks_errors.h
string.o: ./event.h
string.o: ./metafile.h
stroke.o: stroke.c
stroke.o: ./gks_implem.h
stroke.o: ./xgks.h
stroke.o: ./gks_defines.h
stroke.o: ./primitive.h
stroke.o: ./input.h
stroke.o: ./wdt.h
stroke.o: ./wslist.h
stroke.o: ./gkslist.h
stroke.o: ./gks_errors.h
stroke.o: ./event.h
stroke.o: ./metafile.h
text.o: text.c
text.o: ./gks_implem.h
text.o: ./xgks.h
text.o: ./gks_defines.h
text.o: ./primitive.h
text.o: ./input.h
text.o: ./wdt.h
text.o: ./wslist.h
text.o: ./gkslist.h
text.o: ./gks_errors.h
text.o: ./event.h
text.o: ./metafile.h
text.o: ./text.h
transforms.o: transforms.c
transforms.o: ./gks_implem.h
transforms.o: ./xgks.h
transforms.o: ./gks_defines.h
transforms.o: ./primitive.h
transforms.o: ./input.h
transforms.o: ./wdt.h
transforms.o: ./wslist.h
transforms.o: ./gkslist.h
transforms.o: ./gks_errors.h
transforms.o: ./event.h
transforms.o: ./metafile.h
umalloc.o: umalloc.c
update.o: update.c
update.o: ./gks_implem.h
update.o: ./xgks.h
update.o: ./gks_defines.h
update.o: ./primitive.h
update.o: ./input.h
update.o: ./wdt.h
update.o: ./wslist.h
update.o: ./gkslist.h
update.o: ./gks_errors.h
update.o: ./event.h
update.o: ./metafile.h
valuator.o: valuator.c
valuator.o: ./gks_implem.h
valuator.o: ./xgks.h
valuator.o: ./gks_defines.h
valuator.o: ./primitive.h
valuator.o: ./input.h
valuator.o: ./wdt.h
valuator.o: ./wslist.h
valuator.o: ./gkslist.h
valuator.o: ./gks_errors.h
valuator.o: ./event.h
valuator.o: ./metafile.h
