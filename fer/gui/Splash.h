
/*******************************************************************************
       Splash.h
       This header file is included by Splash.c

*******************************************************************************/

#ifndef	_SPLASH_INCLUDED
#define	_SPLASH_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/PushB.h>
#include <Xm/Label.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	Splash;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_Splash( swidget _UxUxParent );

#endif	/* _SPLASH_INCLUDED */
