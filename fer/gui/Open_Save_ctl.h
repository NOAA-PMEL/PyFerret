
/*******************************************************************************
       Open_Save_ctl.h
       This header file is included by Open_Save_ctl.c

*******************************************************************************/

#ifndef	_OPEN_SAVE_CTL_INCLUDED
#define	_OPEN_SAVE_CTL_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/FileSB.h>
#include <X11/Shell.h>

extern Widget	Open_Save_ctl;
extern Widget	fileSelectionBox2;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Open_Save_ctl( swidget _UxUxParent );

#endif	/* _OPEN_SAVE_CTL_INCLUDED */
