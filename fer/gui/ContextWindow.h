
/*******************************************************************************
       ContextWindow.h
       This header file is included by ContextWindow.c

*******************************************************************************/

#ifndef	_CONTEXTWINDOW_INCLUDED
#define	_CONTEXTWINDOW_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/RepType.h>
#include <Xm/DrawingA.h>
#include <Xm/BulletinB.h>
#include <Xm/ToggleB.h>
#include <Xm/ScrollBar.h>
#include <Xm/RowColumn.h>
#include <Xm/Frame.h>
#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	ContextWindow;
extern Widget	frame15;
extern Widget	textField20;
extern Widget	textField21;
extern Widget	textField22;
extern Widget	textField23;
extern Widget	frame12;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_ContextWindow( swidget _UxUxParent );

#endif	/* _CONTEXTWINDOW_INCLUDED */
