
/*******************************************************************************
       Open_Save_jnl.h
       This header file is included by Open_Save_jnl.c

*******************************************************************************/

#ifndef	_OPEN_SAVE_JNL_INCLUDED
#define	_OPEN_SAVE_JNL_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/FileSB.h>
#include <X11/Shell.h>

extern Widget	Save_jnl;
extern Widget	fileSelectionBox4;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Save_jnl( swidget _UxUxParent );

#endif	/* _OPEN_SAVE_JNL_INCLUDED */
