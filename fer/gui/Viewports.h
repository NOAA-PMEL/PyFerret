
/*******************************************************************************
       Viewports.h
       This header file is included by Viewports.c

*******************************************************************************/

#ifndef	_VIEWPORTS_INCLUDED
#define	_VIEWPORTS_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/Label.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	Viewports;
extern Widget	frame14;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Viewports( swidget _UxUxParent );

#endif	/* _VIEWPORTS_INCLUDED */
