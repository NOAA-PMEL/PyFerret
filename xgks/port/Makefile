# Generated automatically from Makefile.in by configure.
# This is the makefile for the Unidata portability package.
#
# $Id$

PACKAGE		= udport
LIBRARY		= udport
MAJOR_NO	= 2
MINOR_NO	= 

CPPFLAGS	= -I. -DNDEBUG
CFLAGS		= -g -O

PORT_CFORTRAN	= @PORT_CFORTRAN@
NEED_FORTC	= yes

PORT_HEADERS	=  stdarg.h stddef.h signal.h stdlib.h string.h
made_headers	= float.h limits.h signal.h stdarg.h stddef.h stdlib.h \
		  string.h time.h
LIBOBJS		= uddummy.o 

prefix		= /home/lotus/kobrien/xgks/lib

SUBDIRS		=  fortc
SUBDIR_TARGETS	= fortc/all fortc/install fortc/clean fortc/distclean
MANIFEST        = COPYRIGHT CUSTOMIZE HISTORY Makefile.in VERSION aclocal.m4 \
		  configure configure.in depend master.mk.in uddummy.c \
		   udposix.h.in stdarg.h.in stddef.h.in fortc.h fortc.fc udalloc.h config.c signal.h.in sigaddset.c sigdelset.c sigemptyset.c sigprocmask.c sigsuspend.c stdlib.h.in atexit.c strerror.c string.h strstr.c string.h.in

# Note that the `all' target might not make `fortc/all'.  That will depend 
# on whether or not that utility is required.

all::		FORCE $(PORT_HEADERS)
	@if [ "$(NEED_FORTC)" = yes ]; then \
	    $(MAKE) $(MFLAGS) fortc/all; \
	else \
	    :; \
	fi
	@$(MAKE) $(MFLAGS) library

# We install the portability library modules directly into an installed
# portability library even though the package using this portability library
# (the client package) probably doesn't reference the installed one.  This is
# done because the client package might be another library and linking
# against it might require some modules of this library.  We don't install
# the portability library just created because different packages can require
# different modules.
#
install::	$(LIBOBJS)
	$(MAKE) $(MFLAGS) $(LIBDIR) LIBDIR=$(LIBDIR)
	$(AR) rcuv $(LIBDIR)/libudport.a $(LIBOBJS)
	$(RANLIB) $(LIBDIR)/libudport.a

clean::		FORCE
	@if [ "$(NEED_FORTC)" = yes ]; then \
	    $(MAKE) $(MFLAGS) fortc/clean; \
	else \
	    :; \
	fi
	rm -f $(made_headers) $(PORT_HEADERS) config

distclean::
	@if [ "$(NEED_FORTC)" = yes ]; then \
	    $(MAKE) $(MFLAGS) fortc/distclean; \
	else \
	    :; \
	fi
	rm -f $(made_headers) $(PORT_HEADERS) config udposix.h master.mk

Makefile:	Makefile.in master.mk.in

float.h:	config
	./config -f > $@
limits.h:	config
	./config -l > $@
signal.h:	signal.h.in
	cp $? $@
stdarg.h:	stdarg.h.in
	cp $? $@
stddef.h:	stddef.h.in
	cp $? $@
stdlib.h:	stdlib.h.in
	cp $? $@
string.h:	string.h.in
	cp $? $@
time.h:		time.h.in
	cp $? $@
cfortran.h:	cfortran_h
	if test "$(PORT_CFORTRAN)" = stdc; then \
	    sed 's/\/\*\*\//##/g' cfortran_h > $@; \
	else \
	    cp cfortran_h $@; \
	fi

# C compiler information program:
#
# NB: config(1) must be compiled unoptimized.  We first assume that `signed
# char' is valid; if that doesn't work, then we assume it's invalid.  If
# that doesn't work, the user will have to determine the appropriate
# compilation flags (see config.c).
#
# If the following program won't compile, then contact
# support@unidata.ucar.edu.
#
config:		config.c
	$(CC) config.c -o config 2> /dev/null || \
	    $(CC) -DNO_SC config.c -o config

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
atexit.o: atexit.c
atexit.o: ./udposix.h
atexit.o: ./stdlib.h
atexit.o: ./udposix.h
config.o: config.c
config.o: ./signal.h
config.o: ./udposix.h
config.o: ./config.c
config.o: ././config.c
date.o: date.c
difftime.o: difftime.c
difftime.o: ./udposix.h
getuduser.o: getuduser.c
getuduser.o: ./udposix.h
getuduser.o: ./stddef.h
getuduser.o: ./udposix.h
getuduser.o: ./stdlib.h
getuduser.o: ./udposix.h
inetutil.o: inetutil.c
inetutil.o: ./string.h
inetutil.o: ./udposix.h
inetutil.o: ./inetutil.h
linklist.o: linklist.c
linklist.o: ./udposix.h
linklist.o: ./string.h
linklist.o: ./udposix.h
linklist.o: ./udalloc.h
linklist.o: ./stddef.h
linklist.o: ./udposix.h
linklist.o: ./stdlib.h
linklist.o: ./udposix.h
linklist.o: ./linklist.h
memmove.o: memmove.c
memmove.o: ./udposix.h
memmove.o: ./string.h
memmove.o: ./udposix.h
pathname.o: pathname.c
pathname.o: ./udposix.h
pathname.o: ./string.h
pathname.o: ./udposix.h
sigaddset.o: sigaddset.c
sigaddset.o: ./udposix.h
sigaddset.o: ./stddef.h
sigaddset.o: ./udposix.h
sigaddset.o: ./signal.h
sigaddset.o: ./udposix.h
sigdelset.o: sigdelset.c
sigdelset.o: ./udposix.h
sigdelset.o: ./stddef.h
sigdelset.o: ./udposix.h
sigdelset.o: ./signal.h
sigdelset.o: ./udposix.h
sigdummy.o: sigdummy.c
sigdummy.o: ./udposix.h
sigdummy.o: ./signal.h
sigdummy.o: ./udposix.h
sigemptyset.o: sigemptyset.c
sigemptyset.o: ./udposix.h
sigemptyset.o: ./stddef.h
sigemptyset.o: ./udposix.h
sigemptyset.o: ./signal.h
sigemptyset.o: ./udposix.h
sigprocmask.o: sigprocmask.c
sigprocmask.o: ./udposix.h
sigprocmask.o: ./stddef.h
sigprocmask.o: ./udposix.h
sigprocmask.o: ./signal.h
sigprocmask.o: ./udposix.h
sigsuspend.o: sigsuspend.c
sigsuspend.o: ./udposix.h
sigsuspend.o: ./signal.h
sigsuspend.o: ./udposix.h
strbuf.o: strbuf.c
strbuf.o: ./udposix.h
strbuf.o: ./string.h
strbuf.o: ./udposix.h
strbuf.o: ./udalloc.h
strbuf.o: ./stddef.h
strbuf.o: ./udposix.h
strbuf.o: ./stdlib.h
strbuf.o: ./udposix.h
strbuf.o: ./strbuf.h
strbuf.o: ./udposix.h
strbuf.o: ./stddef.h
strerror.o: strerror.c
strerror.o: ./udposix.h
strerror.o: ./string.h
strerror.o: ./udposix.h
strftime.o: strftime.c
strftime.o: ./string.h
strftime.o: ./udposix.h
strstr.o: strstr.c
strstr.o: ./udposix.h
strstr.o: ./string.h
strstr.o: ./udposix.h
strvec.o: strvec.c
strvec.o: ./udposix.h
strvec.o: ./stdlib.h
strvec.o: ./udposix.h
strvec.o: ./string.h
strvec.o: ./udposix.h
strvec.o: ./udalloc.h
strvec.o: ./stddef.h
strvec.o: ./udposix.h
strvec.o: ./stdlib.h
strvec.o: ./strvec.h
udalloc.o: udalloc.c
udalloc.o: ./udposix.h
udalloc.o: ./stddef.h
udalloc.o: ./udposix.h
udalloc.o: ./stdlib.h
udalloc.o: ./udposix.h
udalloc.o: ./string.h
udalloc.o: ./udposix.h
udalloc.o: ./udalloc.h
udalloc.o: ./stddef.h
udalloc.o: ./stdlib.h
uddummy.o: uddummy.c
uderrmsg.o: uderrmsg.c
uderrmsg.o: ./udposix.h
uderrmsg.o: ./stdlib.h
uderrmsg.o: ./udposix.h
uderrmsg.o: ./string.h
uderrmsg.o: ./udposix.h
uderrmsg.o: ./stdarg.h
uderrmsg.o: ./udposix.h
uderrmsg.o: ./uderrmsg.h
ulog.o: ulog.c
ulog.o: ./string.h
ulog.o: ./udposix.h
ulog.o: ./signal.h
ulog.o: ./udposix.h
ulog.o: ./inetutil.h
ulog.o: ./ulog.h
