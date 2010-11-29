#
# Makefile (for the FERRET/fer/gui code)
#
# August 11 1997
# Jonathan Callahan (with some help from Joe Sirott)
#

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../platform_specific_flags.mk.$(HOSTTYPE)


#
# non-default suffix rules
#
 
.c.o:
	$(CC) $(CFLAGS) $(GUI_INCLUDES) -c $<

#
# Macros
#

SRCS_F = 

SRCS_C = \
    JC_CallbackUtility.c JC_CommandGen.c JC_InterInterface.c \
	CommandHelp.c CustomLevels.c JC_DefineVariable.c JC_EditDefinedVar.c \
	ErrorLog.c FerretMainWd.c ListManager.c MacroManager.c \
	Open_Save_ctl.c Open_Save_dset.c Open_Save_err.c Open_Save_list.c \
	Open_Save_grd.c Open_Save_jnl.c Open_jnl.c OpenFile.c \
	OpenGO.c OpenGOFile.c PixmapStuff.c PlotOptions.c \
	PrintSetup.c JC_SelectRegridding.c SaveDataObject.c Splash.c \
	JC_Synchronize.c JC_Utility.c Utility.c Viewports.c \
	gui_dummies.c UxXt.c UxMethod.c

OBJS = $(SRCS_C:.c=.o)

LIB    = ../../lib/libgui.a


#
# Targets
#

all:	$(LIB)


$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	$(RANLIB) $@


depend:
	makedepend $(CFLAGS) $(GUI_INCLUDES) $(SRCS_C)
 

clean:
	rm -f core *.o *.a


#
# End of Makefile
#
# DO NOT DELETE
# DO NOT DELETE THIS LINE -- make depend depends on it.

