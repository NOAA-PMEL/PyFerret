
/*******************************************************************************
       ListManager.h
       This header file is included by ListManager.c

*******************************************************************************/

#ifndef	_LISTMANAGER_INCLUDED
#define	_LISTMANAGER_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	ListManager;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_ListManager( swidget _UxUxParent );

#endif	/* _LISTMANAGER_INCLUDED */
