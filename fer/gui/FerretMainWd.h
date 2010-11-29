
/*******************************************************************************
       FerretMainWd.h
       This header file is included by FerretMainWd.c

*******************************************************************************/

#ifndef	_FERRETMAINWD_INCLUDED
#define	_FERRETMAINWD_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/RepType.h>
#include <Xm/DrawingA.h>
#include <Xm/BulletinB.h>
#include <Xm/ScrollBar.h>
#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/Frame.h>
#include <Xm/ToggleB.h>
#include <Xm/CascadeB.h>
#include <Xm/Separator.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	FerretMainWd;
extern Widget	closeDsetButton;
extern Widget	saveButton;
extern Widget	printButton;
extern Widget	defineVariableButton;
extern Widget	editDefinedVarButton;
extern Widget	defineGridButton;
extern Widget	Start_recording2;
extern Widget	macro_pane_LandOutline;
extern Widget	macro_pane_SolidLand;
extern Widget	InfoButton;
extern Widget	ListButton;
extern Widget	SetWindows_SubPane;
extern Widget	CancelWindow_1_Button;
extern Widget	CancelWindow_2_Button;
extern Widget	CancelWindow_3_Button;
extern Widget	CancelWindow_4_Button;
extern Widget	CancelWindow_5_Button;
extern Widget	frame_Data;
extern Widget	rowColumn_Select;
extern Widget	rowColumn_dummy;
extern Widget	frame_context;
extern Widget	toggleButton_Regridding;
extern Widget	label_RegriddingStatus;
extern Widget	frame_plot;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_FerretMainWd( swidget _UxUxParent );

#endif	/* _FERRETMAINWD_INCLUDED */
