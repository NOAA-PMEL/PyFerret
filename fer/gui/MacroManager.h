
/*******************************************************************************
       MacroManager.h
       This header file is included by MacroManager.c

*******************************************************************************/

#ifndef	_MACROMANAGER_INCLUDED
#define	_MACROMANAGER_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/Label.h>
#include <Xm/ToggleB.h>
#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/CascadeB.h>
#include <Xm/Separator.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	MacroManager;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_MacroManager( swidget _UxUxParent );

#endif	/* _MACROMANAGER_INCLUDED */
