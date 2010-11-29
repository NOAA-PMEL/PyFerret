
/*******************************************************************************
       OpenGOFile.h
       This header file is included by OpenGOFile.c

*******************************************************************************/

#ifndef	_OPENGOFILE_INCLUDED
#define	_OPENGOFILE_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/FileSB.h>
#include <X11/Shell.h>

extern Widget	OpenGOFile;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_OpenGOFile( swidget _UxUxParent );

#endif	/* _OPENGOFILE_INCLUDED */
