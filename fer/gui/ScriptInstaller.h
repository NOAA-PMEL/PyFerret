
/*******************************************************************************
       ScriptInstaller.h
       This header file is included by ScriptInstaller.c

*******************************************************************************/

#ifndef	_SCRIPTINSTALLER_INCLUDED
#define	_SCRIPTINSTALLER_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/RowColumn.h>
#include <Xm/List.h>
#include <Xm/PushB.h>
#include <Xm/Label.h>
#include <Xm/Form.h>
#include <Xm/ScrolledW.h>
#include <X11/Shell.h>

extern Widget	ScriptInstaller;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_ScriptInstaller( swidget _UxUxParent );

#endif	/* _SCRIPTINSTALLER_INCLUDED */
