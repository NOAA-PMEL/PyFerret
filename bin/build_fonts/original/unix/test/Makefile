#
# make the sun filters
#
#   @(#)Makefile	1.1   3/10/88
#
DIR= ../../../.
LIBS= $(DIR)/plotlib/plotlib.a $(DIR)/complot/complot.a \
      $(DIR)/ourlib/ourlib.a
DATA= $(DIR)/blkdata.o
SUNVIEW= -lsuntool -lsunwindow -lpixrect -lX11 -lm
CORE= -lcore77 -lcore
FFLAGS= -g -f68881

all:	drwfnt drwmarks drwsym


drwfnt:	drwfnt.o
	f77 $(FFLAGS) -o drwfnt drwfnt.o $(LIBS) $(CORE) $(SUNVIEW)

drwmarks:	drwmarks.o
	f77 $(FFLAGS) -o drwmarks drwmarks.o $(LIBS) $(CORE) $(SUNVIEW)

drwsym:	drwsym.o
	f77 $(FFLAGS) -o drwsym drwsym.o $(LIBS) $(CORE) $(SUNVIEW)

drwnew: drwnew.o
	f77 $(FFLAGS) -o drwnew drwnew.o $(DATA) $(LIBS) $(CORE) $(SUNVIEW)

drwold: drwold.o
	f77 $(FFLAGS) -o drwold drwold.o $(DATA) $(LIBS) $(CORE) $(SUNVIEW)
