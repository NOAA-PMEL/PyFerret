
/*******************************************************************************
       SaveDataObject.h
       This header file is included by SaveDataObject.c

*******************************************************************************/

#ifndef	_SAVEDATAOBJECT_INCLUDED
#define	_SAVEDATAOBJECT_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/Frame.h>
#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	SaveDataObject;
extern Widget	pushButton38;
extern Widget	frame6;
extern Widget	frame10;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_SaveDataObject( swidget _UxUxParent );

#endif	/* _SAVEDATAOBJECT_INCLUDED */
