head	1.1;
access;
symbols;
locks; strict;
comment	@# @;


1.1
date	99.03.02.18.27.21;	author callahan;	state Exp;
branches;
next	;


desc
@revised directory structure now has a top level Makefile
@


1.1
log
@Initial revision
@
text
@#
# Top level Makefile for Ferret External Functions
#
# September 09 1998
# Jonathan Callahan
#
 

#
# Targets
#

all:
	$(MAKE) -i everything "DIRS=`ls -x`"

everything:
	for i in $(DIRS); \
		do \
			(cd $$i; echo "making" $@@ "in $$i..."; \
			$(MAKE) );\
		done

clean:
	$(MAKE) -i clean_everything "DIRS=`ls -x`"

clean_everything:
	for i in $(DIRS); \
		do \
			(cd $$i; echo "making" $@@ "in $$i..."; \
			$(MAKE) clean);\
		done

#
# End of Makefile
#
@
