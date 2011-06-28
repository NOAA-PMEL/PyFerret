
/*******************************************************************************
       PixmapMainWd.h
       This header file is included by PixmapMainWd.c

*******************************************************************************/

#ifndef	_PIXMAPMAINWD_INCLUDED
#define	_PIXMAPMAINWD_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/BulletinB.h>
#include <X11/Shell.h>

extern Widget	PixmapMainWd;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_PixmapMainWd( swidget _UxUxParent );

#endif	/* _PIXMAPMAINWD_INCLUDED */
