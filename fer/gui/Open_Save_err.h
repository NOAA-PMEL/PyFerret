
/*******************************************************************************
       Open_Save_err.h
       This header file is included by Open_Save_err.c

*******************************************************************************/

#ifndef	_OPEN_SAVE_ERR_INCLUDED
#define	_OPEN_SAVE_ERR_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/FileSB.h>
#include <X11/Shell.h>

extern Widget	Open_Save_err;
extern Widget	fileSelectionBox8;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Open_Save_err( swidget _UxUxParent );

#endif	/* _OPEN_SAVE_ERR_INCLUDED */
