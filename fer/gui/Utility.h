
/*******************************************************************************
       Utility.h
       This header file is included by Utility.c

*******************************************************************************/

#ifndef	_UTILITY_INCLUDED
#define	_UTILITY_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <X11/Shell.h>

extern Widget	Utility;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Utility( swidget _UxUxParent );

#endif	/* _UTILITY_INCLUDED */
