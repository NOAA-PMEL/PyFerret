
/*******************************************************************************
       CustomLevels.h
       This header file is included by CustomLevels.c

*******************************************************************************/

#ifndef	_CUSTOMLEVELS_INCLUDED
#define	_CUSTOMLEVELS_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/RepType.h>
#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/TextF.h>
#include <Xm/BulletinB.h>
#include <Xm/Label.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	CustomLevels;
extern Widget	frame7;
extern Widget	textField34;
extern Widget	textField36;
extern Widget	textField37;
extern Widget	optionMenu_p13;
extern Widget	optionMenu_p_b19;
extern Widget	textField35;
extern Widget	textField38;
extern Widget	textField39;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_CustomLevels( swidget _UxUxParent );

#endif	/* _CUSTOMLEVELS_INCLUDED */
