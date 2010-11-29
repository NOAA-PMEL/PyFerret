
/*******************************************************************************
       OpenFile.h
       This header file is included by OpenFile.c

*******************************************************************************/

#ifndef	_OPENFILE_INCLUDED
#define	_OPENFILE_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/List.h>
#include <Xm/ScrolledW.h>
#include <Xm/TextF.h>
#include <Xm/PushB.h>
#include <Xm/Label.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	OpenFile;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_OpenFile( swidget _UxUxParent );

#endif	/* _OPENFILE_INCLUDED */
