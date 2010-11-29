
/*******************************************************************************
       PixmapStuff.h
       This header file is included by PixmapStuff.c

*******************************************************************************/

#ifndef	_PIXMAPSTUFF_INCLUDED
#define	_PIXMAPSTUFF_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <X11/Shell.h>

extern Widget	PixmapStuff;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_PixmapStuff( swidget _UxUxParent );

#endif	/* _PIXMAPSTUFF_INCLUDED */
