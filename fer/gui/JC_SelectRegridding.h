
/*******************************************************************************
       JC_SelectRegridding.h
       This header file is included by JC_SelectRegridding.c

*******************************************************************************/

#ifndef	_JC_SELECTREGRIDDING_INCLUDED
#define	_JC_SELECTREGRIDDING_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/Separator.h>
#include <Xm/TextF.h>
#include <Xm/RowColumn.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	JC_SelectRegridding;
extern Widget	rowColumn_Select_G;
extern Widget	rowColumn_Select_GX;
extern Widget	rowColumn_Select_GY;
extern Widget	rowColumn_Select_GZ;
extern Widget	rowColumn_Select_GT;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_JC_SelectRegridding( swidget _UxUxParent );

#endif	/* _JC_SELECTREGRIDDING_INCLUDED */
