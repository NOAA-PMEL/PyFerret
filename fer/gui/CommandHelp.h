
/*******************************************************************************
       CommandHelp.h
       This header file is included by CommandHelp.c

*******************************************************************************/

#ifndef	_COMMANDHELP_INCLUDED
#define	_COMMANDHELP_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/Text.h>
#include <Xm/Form.h>
#include <Xm/ScrolledW.h>
#include <X11/Shell.h>

extern Widget	CommandHelp;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_CommandHelp( swidget _UxUxParent );

#endif	/* _COMMANDHELP_INCLUDED */
