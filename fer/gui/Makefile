#
# Makefile (for the Ferret GUI code)
#
# December 13 1995
# Jonathan Callahan
#
# Each platform for which code is compiled has its own directory as specified
# by $FER_LINK_DIR.   This directory contains links to the official *.h and *.c
# code.  It also houses the *.o files for that platform.
#
# From now on it should also house a platform_specific_flags.mk file which
# specifies CC, CCFLAGS and RANLIB for the specific platform.
#


include platform_specific_flags.mk.$(HOSTTYPE)		# for CC, CCFLAGS, RANLIB


C_FILES = \
	ferret.c \
	JC_CallbackUtility.c \
	JC_CommandGen.c \
	JC_InterInterface.c \
	CommandHelp.c \
	CustomLevels.c \
	JC_DefineVariable.c \
	JC_EditDefinedVar.c \
	ErrorLog.c \
	FerretMainWd.c \
	ListManager.c \
	MacroManager.c \
	Open_Save_ctl.c \
	Open_Save_dset.c \
	Open_Save_err.c \
	Open_Save_list.c \
	Open_Save_grd.c \
	Open_Save_jnl.c \
	Open_jnl.c \
	OpenFile.c \
	OpenGO.c \
	OpenGOFile.c \
	PixmapStuff.c \
	PlotOptions.c \
	PrintSetup.c \
	JC_SelectRegridding.c \
	SaveDataObject.c \
	Splash.c \
	JC_Synchronize.c \
	JC_Utility.c \
	Utility.c \
	Viewports.c \
	gui_dummies.c \
	UxXt.c \
	UxMethod.c

C_OBJECTS = ${C_FILES:.c=.o}


.c.o: 
	echo ""
	echo "compiling $<"
	${CC} -c ${CCFLAGS} $<


libgui.a: ${C_OBJECTS}
	echo ""
	mv libgui.a libgui.a~
	ar r libgui.a ${C_OBJECTS}
	${RANLIB} libgui.a



#~~~~~~~~~~~~~~~~~~~~ END OF MAKEFILE ~~~~~~~~~~~~~~~~~~~~#



