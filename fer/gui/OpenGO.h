
/*******************************************************************************
       OpenGO.h
       This header file is included by OpenGO.c

*******************************************************************************/

#ifndef	_OPENGO_INCLUDED
#define	_OPENGO_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/List.h>
#include <Xm/ScrolledW.h>
#include <Xm/TextF.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Label.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	OpenGO;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_OpenGO( swidget _UxUxParent );

#endif	/* _OPENGO_INCLUDED */
