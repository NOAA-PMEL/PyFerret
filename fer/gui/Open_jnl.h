
/*******************************************************************************
       Open_jnl.h
       This header file is included by Open_jnl.c

*******************************************************************************/

#ifndef	_OPEN_JNL_INCLUDED
#define	_OPEN_JNL_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/FileSB.h>
#include <X11/Shell.h>

extern Widget	Open_jnl;
extern Widget	fileSelectionBox5;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Open_jnl( swidget _UxUxParent );

#endif	/* _OPEN_JNL_INCLUDED */
