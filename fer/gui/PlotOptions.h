
/*******************************************************************************
       PlotOptions.h
       This header file is included by PlotOptions.c

*******************************************************************************/

#ifndef	_PLOTOPTIONS_INCLUDED
#define	_PLOTOPTIONS_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/RepType.h>
#include <Xm/PushB.h>
#include <Xm/TextF.h>
#include <Xm/RowColumn.h>
#include <Xm/ToggleB.h>
#include <Xm/Label.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	PlotOptions;
extern Widget	frame_VectorOptions;
extern Widget	frame_2DOptions;
extern Widget	textField_SCFLow;
extern Widget	textField_SCFHigh;
extern Widget	textField_SCFDelta;
extern Widget	frame_1DOptions;
extern Widget	lineStyle1_b9;
extern Widget	optionMenu_LSLineStyle;
extern Widget	optionMenu_LSSymbol;
extern Widget	lineStyle1_b1;
extern Widget	optionMenu_LSThickness;
extern Widget	optionMenu18;
extern Widget	frame_SelectOptions;
extern Widget	frame_GeneralOptions;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_PlotOptions( swidget _UxUxParent );

#endif	/* _PLOTOPTIONS_INCLUDED */
