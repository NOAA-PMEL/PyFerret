
/*******************************************************************************
       ErrorLog.h
       This header file is included by ErrorLog.c

*******************************************************************************/

#ifndef	_ERRORLOG_INCLUDED
#define	_ERRORLOG_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	ErrorLog;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_ErrorLog( swidget _UxUxParent );

#endif	/* _ERRORLOG_INCLUDED */
