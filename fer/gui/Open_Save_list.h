
/*******************************************************************************
       Open_Save_list.h
       This header file is included by Open_Save_list.c

*******************************************************************************/

#ifndef	_OPEN_SAVE_LIST_INCLUDED
#define	_OPEN_SAVE_LIST_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/FileSB.h>
#include <X11/Shell.h>

extern Widget	Open_Save_list;
extern Widget	fileSelectionBox6;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Open_Save_list( swidget _UxUxParent );

#endif	/* _OPEN_SAVE_LIST_INCLUDED */
