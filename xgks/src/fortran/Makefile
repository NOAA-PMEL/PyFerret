# Generated automatically from Makefile.in by configure.
# Makefile for the FORTRAN API to the XGKS library
#
# $Id$

LIBNAME		= xgks
REMOTE_LIBRARY	= ../lib/lib$(LIBNAME).a

CPP_X11		= -I/usr/include
INCLUDES	= -I../../port -I../lib $(CPP_X11)
DEFINES		= -DOS_$(OS)
CPPFLAGS	= $(INCLUDES) $(DEFINES) -DNDEBUG
CFLAGS		= -g -O

headers		= f_errno.h fortmac.h fortxgks.h pdrutils.h ps2aixdefs.h
SRCS		= control.fc error.fc escapes.fc fexterninit.c \
		  finqpixel.fc fxgksvers.c gerhnd.fc getlunname.fc \
		  inputevent.fc inputinit.fc inputmode.fc inputreq.fc \
		  inputsamp.fc inqerrlist.fc inqgksdesc.fc inqgkslist.fc \
		  inqlun.f inqseglist.fc inqstate.fc inqwsdesc.fc \
		  inqwslist.fc meta.fc output.fc rep.fc segattr.fc \
		  segment.fc transform.fc utils.fc wsioutattr.fc
LIBOBJS		= control.o error.o escapes.o fexterninit.o finqpixel.o \
		  fxgksvers.o gerhnd.o getlunname.o inputevent.o \
		  inputinit.o inputmode.o inputreq.o inputsamp.o \
		  inqerrlist.o inqgksdesc.o inqgkslist.o inqlun.o \
		  inqseglist.o inqstate.o inqwsdesc.o inqwslist.o meta.o \
		  output.o rep.o segattr.o segment.o transform.o utils.o \
		  wsioutattr.o
MANIFEST	= Makefile.in depend $(HEADERS) $(headers) $(SRCS)
GARBAGE		= control.c error.c escapes.c \
		  finqpixel.c gerhnd.c getlunname.c \
		  inputevent.c inputinit.c inputmode.c inputreq.c \
		  inputsamp.c inqerrlist.c inqgksdesc.c inqgkslist.c \
		  inqseglist.c inqstate.c inqwsdesc.c \
		  inqwslist.c meta.c output.c rep.c segattr.c \
		  segment.c transform.c utils.c wsioutattr.c

all::		$(REMOTE_LIBRARY)

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
control.o: control.c
control.o: ../lib/xgks.h
control.o: ../lib/gks_defines.h
control.o: ./fortxgks.h
control.o: ./fortmac.h
error.o: error.c
error.o: ../lib/xgks.h
error.o: ./fortxgks.h
error.o: ./fortmac.h
escapes.o: escapes.c
escapes.o: ../lib/xgks.h
escapes.o: ./fortxgks.h
escapes.o: ./fortmac.h
fexterninit.o: fexterninit.c
fexterninit.o: ../lib/xgks.h
finqpixel.o: finqpixel.c
finqpixel.o: ../lib/xgks.h
finqpixel.o: ./fortxgks.h
finqpixel.o: ./fortmac.h
fxgksvers.o: fxgksvers.c
gerhnd.o: gerhnd.c
gerhnd.o: ../lib/xgks.h
gerhnd.o: ./fortxgks.h
gerhnd.o: ./fortmac.h
getlunname.o: getlunname.c
getlunname.o: ./fortxgks.h
getlunname.o: ./fortmac.h
inputevent.o: inputevent.c
inputevent.o: ../lib/xgks.h
inputevent.o: ./fortxgks.h
inputevent.o: ./fortmac.h
inputinit.o: inputinit.c
inputinit.o: ../lib/xgks.h
inputinit.o: ./fortxgks.h
inputinit.o: ./fortmac.h
inputinit.o: ./pdrutils.h
inputmode.o: inputmode.c
inputmode.o: ../lib/xgks.h
inputmode.o: ./fortxgks.h
inputmode.o: ./fortmac.h
inputreq.o: inputreq.c
inputreq.o: ../lib/xgks.h
inputreq.o: ./fortxgks.h
inputreq.o: ./fortmac.h
inputsamp.o: inputsamp.c
inputsamp.o: ../lib/xgks.h
inputsamp.o: ./fortxgks.h
inputsamp.o: ./fortmac.h
inqerrlist.o: inqerrlist.c
inqerrlist.o: ../lib/xgks.h
inqerrlist.o: ./fortxgks.h
inqerrlist.o: ./fortmac.h
inqgksdesc.o: inqgksdesc.c
inqgksdesc.o: ../lib/xgks.h
inqgksdesc.o: ./fortxgks.h
inqgksdesc.o: ./fortmac.h
inqgkslist.o: inqgkslist.c
inqgkslist.o: ../lib/xgks.h
inqgkslist.o: ./fortxgks.h
inqgkslist.o: ./fortmac.h
inqseglist.o: inqseglist.c
inqseglist.o: ../lib/xgks.h
inqseglist.o: ./fortxgks.h
inqseglist.o: ./fortmac.h
inqstate.o: inqstate.c
inqstate.o: ../lib/xgks.h
inqstate.o: ./fortxgks.h
inqstate.o: ./fortmac.h
inqwsdesc.o: inqwsdesc.c
inqwsdesc.o: ../lib/xgks.h
inqwsdesc.o: ./fortxgks.h
inqwsdesc.o: ./fortmac.h
inqwslist.o: inqwslist.c
inqwslist.o: ../lib/xgks.h
inqwslist.o: ./fortxgks.h
inqwslist.o: ./fortmac.h
meta.o: meta.c
meta.o: ../lib/xgks.h
meta.o: ./fortxgks.h
meta.o: ./fortmac.h
output.o: output.c
output.o: ../lib/xgks.h
output.o: ./fortxgks.h
output.o: ./fortmac.h
rep.o: rep.c
rep.o: ../lib/xgks.h
rep.o: ./fortxgks.h
rep.o: ./fortmac.h
segattr.o: segattr.c
segattr.o: ../lib/xgks.h
segattr.o: ./fortxgks.h
segattr.o: ./fortmac.h
segment.o: segment.c
segment.o: ../lib/xgks.h
segment.o: ./fortxgks.h
segment.o: ./fortmac.h
transform.o: transform.c
transform.o: ../lib/xgks.h
transform.o: ./fortxgks.h
transform.o: ./fortmac.h
utils.o: utils.c
utils.o: ../lib/xgks.h
utils.o: ./fortxgks.h
utils.o: ./fortmac.h
utils.o: ./pdrutils.h
wsioutattr.o: wsioutattr.c
wsioutattr.o: ../lib/xgks.h
wsioutattr.o: ./fortxgks.h
wsioutattr.o: ./fortmac.h
