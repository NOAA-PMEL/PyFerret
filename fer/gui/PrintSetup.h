
/*******************************************************************************
       PrintSetup.h
       This header file is included by PrintSetup.c

*******************************************************************************/

#ifndef	_PRINTSETUP_INCLUDED
#define	_PRINTSETUP_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/RowColumn.h>
#include <Xm/Frame.h>
#include <Xm/ToggleB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	PrintSetup;
extern Widget	pushButton10;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_PrintSetup( swidget _UxUxParent );

#endif	/* _PRINTSETUP_INCLUDED */
