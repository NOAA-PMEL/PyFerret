/*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*
*/




/*******************************************************************************
	FerretMainWd.c

       Associated Header file: FerretMainWd.h
*******************************************************************************/

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

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "main.h"
#include <Xm/Protocols.h>
#include <X11/Shell.h>

static char stippleBits[] = {
   0x01, 0x00};


static	Widget	form1;
static	Widget	Ferret_app_menu1;
static	Widget	filePane;
static	Widget	openDsetButton;
static	Widget	separator1;
static	Widget	printSetupButton;
static	Widget	separator2;
static	Widget	quitButton;
static	Widget	fileCascade;
static	Widget	editPane;
static	Widget	defineVariable_SubPane;
static	Widget	DV_aVplusb;
static	Widget	DV_aUplusbV;
static	Widget	DV_funcV;
static	Widget	editCascade;
static	Widget	scriptsPane;
static	Widget	Manage1;
static	Widget	macro_pane_b1;
static	Widget	macro_pane_b2;
static	Widget	scriptsCascade;
static	Widget	helpPane;
static	Widget	AboutFerretButton;
static	Widget	helpCascade;
static	Widget	viewPane;
static	Widget	FerretOutputButton;
static	Widget	errorlog;
static	Widget	viewCascade;
static	Widget	optionsPane;
static	Widget	PlotOptions_Button;
static	Widget	Viewports_Button;
static	Widget	SetWindow_1_Button;
static	Widget	SetWindow_2_Button;
static	Widget	SetWindow_3_Button;
static	Widget	SetWindow_4_Button;
static	Widget	SetWindow_5_Button;
static	Widget	SetWindow_Cascade;
static	Widget	CancelWindows_SubPane;
static	Widget	CancelWindow_Cascade;
static	Widget	Separator;
static	Widget	IncludeHours_Button;
static	Widget	ShowMap_Button;
static	Widget	optionsCascade;
static	Widget	debugPane;
static	Widget	PrintAxis_SubPane;
static	Widget	X_AxisButton;
static	Widget	Y_AxisButton;
static	Widget	Z_AxisButton;
static	Widget	T_AxisButton;
static	Widget	PrintAxisButton;
static	Widget	PrintSpan_SubPane;
static	Widget	X_SpanButton;
static	Widget	Y_SpanButton;
static	Widget	Z_SpanButton;
static	Widget	T_SpanButton;
static	Widget	PrintSpanButton;
static	Widget	debugCascade;
static	Widget	label_Data;
static	Widget	form_Data;
static	Widget	pushButton_dummy;
static	Widget	rowColumn_Data;
static	Widget	pushButton_Clone;
static	Widget	textField_Variable;
static	Widget	textField_Dataset;
static	Widget	label1;
static	Widget	label_DataFrameStatus;
static	Widget	label65;
static	Widget	form_Context;
static	Widget	form_Transforms;
static	Widget	label_Transform;
static	Widget	label_Argument;
static	Widget	rowColumn_TRANS_X;
static	Widget	optionMenu_5;
static	Widget	optionMenu_p_b8;
static	Widget	optionMenu_2_b91;
static	Widget	optionMenu_2_b92;
static	Widget	optionMenu_2_b93;
static	Widget	optionMenu_2_b94;
static	Widget	optionMenu_2_b95;
static	Widget	optionMenu_2_b96;
static	Widget	optionMenu_2_b97;
static	Widget	optionMenu_2_b98;
static	Widget	optionMenu_2_b99;
static	Widget	optionMenu_2_b100;
static	Widget	optionMenu_2_b101;
static	Widget	optionMenu_2_b102;
static	Widget	optionMenu_2_b103;
static	Widget	optionMenu_2_b104;
static	Widget	optionMenu_2_b105;
static	Widget	optionMenu_2_b106;
static	Widget	optionMenu_2_b107;
static	Widget	optionMenu_2_b108;
static	Widget	optionMenu_2_b109;
static	Widget	optionMenu_2_b110;
static	Widget	optionMenu_2_b111;
static	Widget	optionMenu_2_b112;
static	Widget	optionMenu_2_b113;
static	Widget	optionMenu_2_b114;
static	Widget	optionMenu_X_TRANS;
static	Widget	rowColumn_SubTrans;
static	Widget	textField_X_ARG;
static	Widget	rowColumn_TRANS_Y;
static	Widget	optionMenu_1;
static	Widget	optionMenu_p_b9;
static	Widget	optionMenu_2_b1;
static	Widget	optionMenu_2_b6;
static	Widget	optionMenu_2_b25;
static	Widget	optionMenu_2_b26;
static	Widget	optionMenu_2_b27;
static	Widget	optionMenu_2_b28;
static	Widget	optionMenu_2_b29;
static	Widget	optionMenu_2_b30;
static	Widget	optionMenu_2_b31;
static	Widget	optionMenu_2_b32;
static	Widget	optionMenu_2_b33;
static	Widget	optionMenu_2_b34;
static	Widget	optionMenu_2_b35;
static	Widget	optionMenu_2_b36;
static	Widget	optionMenu_2_b37;
static	Widget	optionMenu_2_b38;
static	Widget	optionMenu_2_b39;
static	Widget	optionMenu_2_b40;
static	Widget	optionMenu_2_b41;
static	Widget	optionMenu_1_b1;
static	Widget	optionMenu_2_b42;
static	Widget	optionMenu_2_b43;
static	Widget	optionMenu_2_b44;
static	Widget	optionMenu_1_b2;
static	Widget	optionMenu_Y_TRANS;
static	Widget	rowColumn_SubTrans1;
static	Widget	textField_Y_ARG;
static	Widget	rowColumn_TRANS_Z;
static	Widget	optionMenu_2;
static	Widget	optionMenu_p_b2;
static	Widget	optionMenu_2_b2;
static	Widget	optionMenu_2_b3;
static	Widget	optionMenu_2_b4;
static	Widget	optionMenu_2_b5;
static	Widget	optionMenu_2_b7;
static	Widget	optionMenu_2_b8;
static	Widget	optionMenu_2_b9;
static	Widget	optionMenu_2_b10;
static	Widget	optionMenu_2_b11;
static	Widget	optionMenu_2_b12;
static	Widget	optionMenu_2_b13;
static	Widget	optionMenu_2_b14;
static	Widget	optionMenu_2_b15;
static	Widget	optionMenu_2_b16;
static	Widget	optionMenu_2_b17;
static	Widget	optionMenu_2_b18;
static	Widget	optionMenu_2_b19;
static	Widget	optionMenu_2_b20;
static	Widget	optionMenu_2_b21;
static	Widget	optionMenu_3_b1;
static	Widget	optionMenu_2_b22;
static	Widget	optionMenu_2_b23;
static	Widget	optionMenu_2_b24;
static	Widget	optionMenu_3_b2;
static	Widget	optionMenu_Z_TRANS;
static	Widget	rowColumn_SubTrans2;
static	Widget	textField_Z_ARG;
static	Widget	rowColumn_TRANS_T;
static	Widget	optionMenu_3;
static	Widget	optionMenu_p_b12;
static	Widget	optionMenu_2_b45;
static	Widget	optionMenu_2_b46;
static	Widget	optionMenu_2_b47;
static	Widget	optionMenu_2_b48;
static	Widget	optionMenu_2_b49;
static	Widget	optionMenu_2_b50;
static	Widget	optionMenu_2_b51;
static	Widget	optionMenu_2_b52;
static	Widget	optionMenu_2_b53;
static	Widget	optionMenu_2_b54;
static	Widget	optionMenu_2_b55;
static	Widget	optionMenu_2_b56;
static	Widget	optionMenu_2_b57;
static	Widget	optionMenu_2_b58;
static	Widget	optionMenu_2_b59;
static	Widget	optionMenu_2_b60;
static	Widget	optionMenu_2_b61;
static	Widget	optionMenu_2_b62;
static	Widget	optionMenu_2_b63;
static	Widget	optionMenu_4_b1;
static	Widget	optionMenu_2_b64;
static	Widget	optionMenu_2_b65;
static	Widget	optionMenu_2_b66;
static	Widget	optionMenu_4_b2;
static	Widget	optionMenu_T_TRANS;
static	Widget	rowColumn_SubTrans3;
static	Widget	textField_T_ARG;
static	Widget	form_XYZT;
static	Widget	rowColumn_XYZT_Regridding;
static	Widget	rowColumn_subRegridding;
static	Widget	pushButton_SelectRegridding;
static	Widget	rowColumn_XYZT_X;
static	Widget	toggleButton_X;
static	Widget	optionMenu_Xp;
static	Widget	optionMenu_Xp_Longitude;
static	Widget	optionMenu_Xp_Index;
static	Widget	optionMenu_X;
static	Widget	rowColumn_XYZT_Y;
static	Widget	toggleButton_Y;
static	Widget	optionMenu_Yp;
static	Widget	optionMenu_Yp_Latitude;
static	Widget	optionMenu_Yp_Index;
static	Widget	optionMenu_Y;
static	Widget	rowColumn_XYZT_Z;
static	Widget	toggleButton_Z;
static	Widget	optionMenu_Zp;
static	Widget	optionMenu_Zp_Depth;
static	Widget	optionMenu_Zp_Index;
static	Widget	optionMenu_Z;
static	Widget	rowColumn_XYZT_T;
static	Widget	toggleButton_T;
static	Widget	optionMenu_Tp;
static	Widget	optionMenu_Tp_Calendar;
static	Widget	optionMenu_Tp_Climatology;
static	Widget	optionMenu_Tp_Model;
static	Widget	optionMenu_Tp_Index;
static	Widget	optionMenu_T;
static	Widget	form_Region;
static	Widget	textField_X_LO;
static	Widget	textField_Y_LO;
static	Widget	textField_Z_LO;
static	Widget	scrollBar_X_LO;
static	Widget	scrollBar_Y_LO;
static	Widget	scrollBar_Z_LO;
static	Widget	textField_T_LO;
static	Widget	scrollBar_T_LO;
static	Widget	textField_X_PT;
static	Widget	textField_Y_PT;
static	Widget	textField_Z_PT;
static	Widget	scrollBar_X_PT;
static	Widget	scrollBar_Y_PT;
static	Widget	scrollBar_Z_PT;
static	Widget	textField_T_PT;
static	Widget	scrollBar_T_PT;
static	Widget	textField_X_HI;
static	Widget	textField_Y_HI;
static	Widget	textField_Z_HI;
static	Widget	scrollBar_Z_HI;
static	Widget	scrollBar_Y_HI;
static	Widget	scrollBar_X_HI;
static	Widget	textField_T_HI;
static	Widget	scrollBar_T_HI;
static	Widget	optionMenu_p4;
static	Widget	optionMenu_p_b4;
static	Widget	optionMenu_p4_b2;
static	Widget	optionMenu_p4_b3;
static	Widget	optionMenu_p4_b4;
static	Widget	optionMenu_p4_b5;
static	Widget	optionMenu_p4_b6;
static	Widget	optionMenu_p4_b7;
static	Widget	optionMenu_p4_b8;
static	Widget	optionMenu_p4_b9;
static	Widget	optionMenu_p4_b10;
static	Widget	optionMenu_p4_b11;
static	Widget	optionMenu_p4_b12;
static	Widget	optionMenu_p4_b13;
static	Widget	optionMenu_p4_b14;
static	Widget	optionMenu_p4_b15;
static	Widget	optionMenu_p4_b16;
static	Widget	optionMenu_Geometry;
static	Widget	label10;
static	Widget	form_plot;
static	Widget	rowColumn_PlotRadios;
static	Widget	toggleButton_Line;
static	Widget	toggleButton_Scatter;
static	Widget	toggleButton_Shade;
static	Widget	toggleButton_Contour;
static	Widget	toggleButton_Fill;
static	Widget	toggleButton_Vector;
static	Widget	rowColumn_PlotButtons;
static	Widget	pushButton_Plot;
static	Widget	pushButton_Overlay;
static	Widget	frame_map;
static	Widget	bulletinBoard9;
static	Widget	drawingArea1;
static	Widget	StartupMessage;
static	XtAppContext	app;
static	XGCValues	gcv;
static	XGCValues	drGcv;
static	GC	gc;
static	GC	drGc;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "FerretMainWd.h"
#undef CONTEXT_MACRO_ACCESS

Widget	FerretMainWd;
Widget	closeDsetButton;
Widget	saveButton;
Widget	printButton;
Widget	defineVariableButton;
Widget	editDefinedVarButton;
Widget	defineGridButton;
Widget	Start_recording2;
Widget	macro_pane_LandOutline;
Widget	macro_pane_SolidLand;
Widget	InfoButton;
Widget	ListButton;
Widget	SetWindows_SubPane;
Widget	CancelWindow_1_Button;
Widget	CancelWindow_2_Button;
Widget	CancelWindow_3_Button;
Widget	CancelWindow_4_Button;
Widget	CancelWindow_5_Button;
Widget	frame_Data;
Widget	rowColumn_Select;
Widget	rowColumn_dummy;
Widget	frame_context;
Widget	toggleButton_Regridding;
Widget	label_RegriddingStatus;
Widget	frame_plot;

/*******************************************************************************
       The following are translation tables.
*******************************************************************************/

static char	*Map_translations = "#replace\n\
<Btn1Down>:Map_Btn1Down()\n\
<Btn1Up>:Map_Btn1Up()\n\
<Btn1Motion>:Map_Btn1Motion()\n";

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#include "main.c"

/*******************************************************************************
       The following are Action functions.
*******************************************************************************/

static	void	action_Map_Btn1Motion(
			Widget wgt, 
			XEvent *ev, 
			String *parm, 
			Cardinal *p_UxNumParams)
{
	Cardinal		UxNumParams = *p_UxNumParams;
	Widget                  UxWidget = wgt;
	XEvent                  *UxEvent = ev;
	String                  *UxParams = parm;
	{
		JC_Map_Motion1Notify_Action(UxWidget, UxEvent, UxParams, UxNumParams);
	}
}

static	void	action_Map_Btn1Up(
			Widget wgt, 
			XEvent *ev, 
			String *parm, 
			Cardinal *p_UxNumParams)
{
	Cardinal		UxNumParams = *p_UxNumParams;
	Widget                  UxWidget = wgt;
	XEvent                  *UxEvent = ev;
	String                  *UxParams = parm;
	{
		JC_Map_Button1Release_Action(UxWidget, UxEvent, UxParams, UxNumParams);
	}
}

static	void	action_Map_Btn1Down(
			Widget wgt, 
			XEvent *ev, 
			String *parm, 
			Cardinal *p_UxNumParams)
{
	Cardinal		UxNumParams = *p_UxNumParams;
	Widget                  UxWidget = wgt;
	XEvent                  *UxEvent = ev;
	String                  *UxParams = parm;
	{
		JC_Map_Button1Press_Action(UxWidget, UxEvent, UxParams, UxNumParams);
	}
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	activateCB_openDsetButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
		OpenFile = create_OpenFile(NO_PARENT);
	
		/* popup Open file */
		XtPopup(OpenFile, XtGrabNone);
		XtVaSetValues(OpenFile,
			XmNiconic, False,
			NULL);
	}
}

static	void	activateCB_printButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	PrintCmdCB() ;
}

static	void	activateCB_printSetupButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	PrintSetup = create_PrintSetup(NO_PARENT);
	XtVaSetValues(PrintSetup,
		XmNiconic, False,
		NULL);
	InitPS();
}

static	void	activateCB_quitButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	quit_cb(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_DV_aVplusb(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget JC_DefineVariable;
	
	JC_DefineVariable = create_JC_DefineVariable(NO_PARENT, FUNC_PLUS_CONSTANT);
	}
}

static	void	activateCB_DV_aUplusbV(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget JC_DefineVariable;
	
	JC_DefineVariable = create_JC_DefineVariable(NO_PARENT, FUNC_LINEAR_COMBINATION);
	}
}

static	void	activateCB_DV_funcV(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget JC_DefineVariable;
	
	JC_DefineVariable = create_JC_DefineVariable(NO_PARENT, FUNC_FUNCTION1);
	}
}

static	void	activateCB_Start_recording2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
		XmString buttonLabel, startLabel, endLabel;
	
		startLabel = XmStringCreate("Start Recording", XmFONTLIST_DEFAULT_TAG);
		endLabel = XmStringCreate("Stop Recording", XmFONTLIST_DEFAULT_TAG);
	
		/* get the widget's label */
		XtVaGetValues(UxWidget,
			XmNlabelString, &buttonLabel,
			NULL);
	
		if (XmStringCompare(buttonLabel, startLabel)) {
			/* not recording--start*/
			XtVaSetValues(UxWidget,
				XmNlabelString, endLabel,
				NULL);
			gMacroIsRecording = 1;
			if (gMMIsOpen)
				SetRecordBtn();
		}
		else {
			/* recording--stop */
			XtVaSetValues(UxWidget,
				XmNlabelString, startLabel,
				NULL);
			gMacroIsRecording = 0;
			if (gMMIsOpen)
				SetStopBtn();
		}
		XmStringFree(startLabel);
		XmStringFree(endLabel);
		XmStringFree(buttonLabel);
	}
}

static	void	activateCB_Manage1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget MacroManager;
	
	MacroManager = create_MacroManager(NO_PARENT);
	XtVaSetValues(MacroManager,
		XmNiconic, False,
		NULL);
	}
}

static	void	activateCB_macro_pane_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	OpenGO = create_OpenGO(NO_PARENT);
	XtPopup(OpenGO, XtGrabNone);
	XtVaSetValues(OpenGO,
		XmNiconic, False,
		NULL);
}

static	void	activateCB_macro_pane_LandOutline(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{ 
	ferret_command("GO land", IGNORE_COMMAND_WIDGET);
	}
}

static	void	activateCB_macro_pane_SolidLand(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_MainMenu_SolidLand_CB();
	}
}

static	void	activateCB_AboutFerretButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget Splash;
	
	Splash = create_Splash(NO_PARENT);
	XtVaSetValues(Splash,
		XmNiconic, False,
		NULL);
	
	}
}

static	void	activateCB_InfoButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{JC_InfoButton_CB();}
}

static	void	activateCB_ListButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{JC_ListButton_CB();}
}

static	void	activateCB_FerretOutputButton(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ListManager = create_ListManager(NO_PARENT);
	XtVaSetValues(ListManager,
		XmNiconic, False,
		NULL);
	}
}

static	void	activateCB_errorlog(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
		  XtPopup(UxGetWidget(ErrorLog), XtGrabNone);
	}
}

static	void	activateCB_optionMenu_p_b8(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b91(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b92(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b93(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b94(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b95(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b96(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b97(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b98(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b99(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b100(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b101(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b102(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b103(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b104(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b105(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b106(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b107(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b108(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b109(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b110(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTrans = (char *)UxClientData;
	JC_TransMenu_CB(theTrans);
	}
}

static	void	activateCB_optionMenu_2_b111(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b112(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b113(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b114(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *inTrans = (char *)UxClientData;
	JC_TransMenu_CB(inTrans);
	}
}

static	void	valueChangedCB_textField_X_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	
	}
}

static	void	activateCB_textField_X_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	losingFocusCB_textField_X_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	activateCB_optionMenu_p_b9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b6(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b25(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b26(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b27(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b28(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b29(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b30(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b31(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b32(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b33(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b34(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b35(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b36(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b37(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b38(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b39(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b40(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b41(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_1_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *inTrans = (char *)UxClientData;
	JC_TransMenu_CB(inTrans);
	
	}
}

static	void	activateCB_optionMenu_2_b42(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b43(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b44(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_1_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *inTrans = (char *)UxClientData;
	JC_TransMenu_CB(inTrans);
	}
}

static	void	valueChangedCB_textField_Y_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	
	}
}

static	void	activateCB_textField_Y_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	losingFocusCB_textField_Y_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	activateCB_optionMenu_p_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b3(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b4(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b5(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b7(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b8(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b10(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b11(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b12(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b13(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b14(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b15(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b16(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b17(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b18(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b19(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b20(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b21(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_3_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_TransMenu_CB((char *)UxClientData);
	}
}

static	void	activateCB_optionMenu_2_b22(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b23(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b24(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_3_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_TransMenu_CB((char *)UxClientData);
	}
}

static	void	valueChangedCB_textField_Z_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	
	}
}

static	void	activateCB_textField_Z_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	losingFocusCB_textField_Z_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	activateCB_optionMenu_p_b12(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b45(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b46(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b47(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b48(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b49(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b50(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b51(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b52(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b53(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b54(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b55(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b56(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b57(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b58(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b59(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b60(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b61(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b62(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b63(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_4_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_TransMenu_CB((char *)UxClientData);
	}
}

static	void	activateCB_optionMenu_2_b64(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b65(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_2_b66(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	char *theTransform;
	
	theTransform = (char *)UxClientData;
	
	JC_TransMenu_CB(theTransform);
	}
}

static	void	activateCB_optionMenu_4_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_TransMenu_CB((char *)UxClientData);
	}
}

static	void	valueChangedCB_textField_T_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	
	}
}

static	void	activateCB_textField_T_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	losingFocusCB_textField_T_ARG(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_TransArg_CB(UxWidget, UxClientData);
}

static	void	activateCB_pushButton_SelectRegridding(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{create_JC_SelectRegridding(NO_PARENT);}
}

static	void	activateCB_optionMenu_Xp_Longitude(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(X_AXIS, FALSE);
}

static	void	activateCB_optionMenu_Xp_Index(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(X_AXIS, TRUE);
}

static	void	activateCB_optionMenu_Yp_Latitude(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(Y_AXIS, FALSE);
}

static	void	activateCB_optionMenu_Yp_Index(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(Y_AXIS, TRUE);
}

static	void	activateCB_optionMenu_Zp_Depth(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(Z_AXIS, FALSE);
}

static	void	activateCB_optionMenu_Zp_Index(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(Z_AXIS, TRUE);
}

static	void	activateCB_optionMenu_Tp_Calendar(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(T_AXIS, CALENDAR_TIME);
}

static	void	activateCB_optionMenu_Tp_Climatology(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(T_AXIS, CLIMATOLOGY_TIME);
}

static	void	activateCB_optionMenu_Tp_Model(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(T_AXIS, MODEL_TIME);
}

static	void	activateCB_optionMenu_Tp_Index(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_sswwMenu_CB(T_AXIS, INDEX_TIME);
}

static	void	dragCB_scrollBar_X_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_X_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_Y_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_Y_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_Z_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_Z_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_T_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_T_LO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_X_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_X_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_Y_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_Y_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_Z_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_Z_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_T_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_T_PT(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_Z_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_Z_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_Y_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_Y_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_X_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_X_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	dragCB_scrollBar_T_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	valueChangedCB_scrollBar_T_HI(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_scrollBar_CB(UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p_b4(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b3(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b4(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b5(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b6(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b7(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b8(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b10(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b11(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b12(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b13(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b14(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b15(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	activateCB_optionMenu_p4_b16(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_GeometryMenu_CB(UxClientData);
	}
}

static	void	valueChangedCB_toggleButton_Line(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True)
		JC_PlotTypeToggle_CB(PLOT_LINE);
	
	}
}

static	void	valueChangedCB_toggleButton_Scatter(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True)
		JC_PlotTypeToggle_CB(PLOT_SCATTER);
	
	}
}

static	void	valueChangedCB_toggleButton_Shade(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True)
		JC_PlotTypeToggle_CB(PLOT_SHADE);
	
	}
}

static	void	valueChangedCB_toggleButton_Contour(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True)
		JC_PlotTypeToggle_CB(PLOT_CONTOUR);
	
	}
}

static	void	valueChangedCB_toggleButton_Fill(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True)
		JC_PlotTypeToggle_CB(PLOT_FILL);
	
	}
}

static	void	valueChangedCB_toggleButton_Vector(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True)
		JC_PlotTypeToggle_CB(PLOT_VECTOR);
	
	}
}

static	void	activateCB_pushButton_Plot(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_PlotButton_CB();
	}
}

static	void	activateCB_pushButton_Overlay(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_OverlayButton_CB();
	}
}

static	void	exposeCB_drawingArea1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	JC_Map_NewRegion(&GLOBAL_Region);
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_FerretMainWd()
{
	Widget		_UxParent;
	Widget		filePane_shell;
	Widget		editPane_shell;
	Widget		defineVariable_SubPane_shell;
	Widget		scriptsPane_shell;
	Widget		helpPane_shell;
	Widget		viewPane_shell;
	Widget		optionsPane_shell;
	Widget		SetWindows_SubPane_shell;
	Widget		CancelWindows_SubPane_shell;
	Widget		debugPane_shell;
	Widget		PrintAxis_SubPane_shell;
	Widget		PrintSpan_SubPane_shell;
	Widget		optionMenu_5_shell;
	Widget		optionMenu_1_shell;
	Widget		optionMenu_2_shell;
	Widget		optionMenu_3_shell;
	Widget		optionMenu_Xp_shell;
	Widget		optionMenu_Yp_shell;
	Widget		optionMenu_Zp_shell;
	Widget		optionMenu_Tp_shell;
	Widget		optionMenu_p4_shell;


	/* Creation of FerretMainWd */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	FerretMainWd = XtVaCreatePopupShell( "FerretMainWd",
			topLevelShellWidgetClass,
			_UxParent,
			XmNdeleteResponse, XmDO_NOTHING,
			XmNallowShellResize, TRUE,
			XmNiconName, "Ferret: Main Window",
			NULL );


	/* Creation of form1 */
	form1 = XtVaCreateManagedWidget( "form1",
			xmFormWidgetClass,
			FerretMainWd,
			NULL );


	/* Creation of Ferret_app_menu1 */
	Ferret_app_menu1 = XtVaCreateManagedWidget( "Ferret_app_menu1",
			xmRowColumnWidgetClass,
			form1,
			XmNrowColumnType, XmMENU_BAR,
			XmNmenuAccelerator, "<KeyUp>F10",
			XmNrightOffset, 0,
			XmNrightAttachment, XmATTACH_FORM,
			XmNleftOffset, 0,
			XmNpacking, XmPACK_COLUMN,
			XmNleftAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of filePane */
	filePane_shell = XtVaCreatePopupShell ("filePane_shell",
			xmMenuShellWidgetClass, Ferret_app_menu1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	filePane = XtVaCreateWidget( "filePane",
			xmRowColumnWidgetClass,
			filePane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of openDsetButton */
	openDsetButton = XtVaCreateManagedWidget( "openDsetButton",
			xmPushButtonWidgetClass,
			filePane,
			RES_CONVERT( XmNlabelString, "Open Data Set..." ),
			RES_CONVERT( XmNmnemonic, "O" ),
			XmNaccelerator, "<control>o",
			NULL );
	XtAddCallback( openDsetButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_openDsetButton,
		(XtPointer) NULL );



	/* Creation of closeDsetButton */
	closeDsetButton = XtVaCreateManagedWidget( "closeDsetButton",
			xmPushButtonWidgetClass,
			filePane,
			RES_CONVERT( XmNlabelString, "Close This Dataset" ),
			NULL );
	XtAddCallback( closeDsetButton, XmNactivateCallback,
		(XtCallbackProc) JC_CloseDataset_CB,
		(XtPointer) NULL );



	/* Creation of saveButton */
	saveButton = XtVaCreateManagedWidget( "saveButton",
			xmPushButtonWidgetClass,
			filePane,
			RES_CONVERT( XmNlabelString, "Save Data..." ),
			NULL );
	XtAddCallback( saveButton, XmNactivateCallback,
		(XtCallbackProc) JC_SaveButton_CB,
		(XtPointer) NULL );



	/* Creation of separator1 */
	separator1 = XtVaCreateManagedWidget( "separator1",
			xmSeparatorWidgetClass,
			filePane,
			NULL );


	/* Creation of printButton */
	printButton = XtVaCreateManagedWidget( "printButton",
			xmPushButtonWidgetClass,
			filePane,
			RES_CONVERT( XmNlabelString, "Print" ),
			NULL );
	XtAddCallback( printButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_printButton,
		(XtPointer) NULL );



	/* Creation of printSetupButton */
	printSetupButton = XtVaCreateManagedWidget( "printSetupButton",
			xmPushButtonWidgetClass,
			filePane,
			RES_CONVERT( XmNlabelString, "Print Setup..." ),
			NULL );
	XtAddCallback( printSetupButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_printSetupButton,
		(XtPointer) NULL );



	/* Creation of separator2 */
	separator2 = XtVaCreateManagedWidget( "separator2",
			xmSeparatorWidgetClass,
			filePane,
			NULL );


	/* Creation of quitButton */
	quitButton = XtVaCreateManagedWidget( "quitButton",
			xmPushButtonWidgetClass,
			filePane,
			RES_CONVERT( XmNlabelString, "Quit" ),
			RES_CONVERT( XmNmnemonic, "E" ),
			NULL );
	XtAddCallback( quitButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_quitButton,
		(XtPointer) NULL );



	/* Creation of fileCascade */
	fileCascade = XtVaCreateManagedWidget( "fileCascade",
			xmCascadeButtonWidgetClass,
			Ferret_app_menu1,
			RES_CONVERT( XmNlabelString, "File" ),
			RES_CONVERT( XmNmnemonic, "F" ),
			XmNsubMenuId, filePane,
			NULL );


	/* Creation of editPane */
	editPane_shell = XtVaCreatePopupShell ("editPane_shell",
			xmMenuShellWidgetClass, Ferret_app_menu1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	editPane = XtVaCreateWidget( "editPane",
			xmRowColumnWidgetClass,
			editPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of defineVariable_SubPane */
	defineVariable_SubPane_shell = XtVaCreatePopupShell ("defineVariable_SubPane_shell",
			xmMenuShellWidgetClass, editPane,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	defineVariable_SubPane = XtVaCreateWidget( "defineVariable_SubPane",
			xmRowColumnWidgetClass,
			defineVariable_SubPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of DV_aVplusb */
	DV_aVplusb = XtVaCreateManagedWidget( "DV_aVplusb",
			xmPushButtonWidgetClass,
			defineVariable_SubPane,
			RES_CONVERT( XmNlabelString, "(a*V)+b" ),
			NULL );
	XtAddCallback( DV_aVplusb, XmNactivateCallback,
		(XtCallbackProc) activateCB_DV_aVplusb,
		(XtPointer) NULL );



	/* Creation of DV_aUplusbV */
	DV_aUplusbV = XtVaCreateManagedWidget( "DV_aUplusbV",
			xmPushButtonWidgetClass,
			defineVariable_SubPane,
			RES_CONVERT( XmNlabelString, "a*U+b*V" ),
			NULL );
	XtAddCallback( DV_aUplusbV, XmNactivateCallback,
		(XtCallbackProc) activateCB_DV_aUplusbV,
		(XtPointer) NULL );



	/* Creation of DV_funcV */
	DV_funcV = XtVaCreateManagedWidget( "DV_funcV",
			xmPushButtonWidgetClass,
			defineVariable_SubPane,
			RES_CONVERT( XmNlabelString, "f(V)" ),
			NULL );
	XtAddCallback( DV_funcV, XmNactivateCallback,
		(XtCallbackProc) activateCB_DV_funcV,
		(XtPointer) NULL );



	/* Creation of defineVariableButton */
	defineVariableButton = XtVaCreateManagedWidget( "defineVariableButton",
			xmCascadeButtonWidgetClass,
			editPane,
			RES_CONVERT( XmNlabelString, "Define Variable" ),
			XmNsubMenuId, defineVariable_SubPane,
			NULL );


	/* Creation of editDefinedVarButton */
	editDefinedVarButton = XtVaCreateManagedWidget( "editDefinedVarButton",
			xmPushButtonWidgetClass,
			editPane,
			RES_CONVERT( XmNlabelString, "Edit Defined Var ..." ),
			NULL );
	XtAddCallback( editDefinedVarButton, XmNactivateCallback,
		(XtCallbackProc) JC_EditDefinedVar_CB,
		(XtPointer) NULL );



	/* Creation of defineGridButton */
	defineGridButton = XtVaCreateManagedWidget( "defineGridButton",
			xmPushButtonWidgetClass,
			editPane,
			RES_CONVERT( XmNlabelString, "Define Grid ..." ),
			NULL );
	XtAddCallback( defineGridButton, XmNactivateCallback,
		(XtCallbackProc) JC_DefineGrid_CB,
		(XtPointer) NULL );



	/* Creation of editCascade */
	editCascade = XtVaCreateManagedWidget( "editCascade",
			xmCascadeButtonWidgetClass,
			Ferret_app_menu1,
			RES_CONVERT( XmNlabelString, "Edit" ),
			RES_CONVERT( XmNmnemonic, "E" ),
			XmNsubMenuId, editPane,
			NULL );


	/* Creation of scriptsPane */
	scriptsPane_shell = XtVaCreatePopupShell ("scriptsPane_shell",
			xmMenuShellWidgetClass, Ferret_app_menu1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	scriptsPane = XtVaCreateWidget( "scriptsPane",
			xmRowColumnWidgetClass,
			scriptsPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNtearOffModel, XmTEAR_OFF_ENABLED,
			NULL );


	/* Creation of Start_recording2 */
	Start_recording2 = XtVaCreateManagedWidget( "Start_recording2",
			xmPushButtonWidgetClass,
			scriptsPane,
			RES_CONVERT( XmNlabelString, "Start Recording" ),
			RES_CONVERT( XmNmnemonic, "R" ),
			NULL );
	XtAddCallback( Start_recording2, XmNactivateCallback,
		(XtCallbackProc) activateCB_Start_recording2,
		(XtPointer) NULL );



	/* Creation of Manage1 */
	Manage1 = XtVaCreateManagedWidget( "Manage1",
			xmPushButtonWidgetClass,
			scriptsPane,
			RES_CONVERT( XmNlabelString, "Command Line..." ),
			RES_CONVERT( XmNmnemonic, "M" ),
			NULL );
	XtAddCallback( Manage1, XmNactivateCallback,
		(XtCallbackProc) activateCB_Manage1,
		(XtPointer) NULL );



	/* Creation of macro_pane_b1 */
	macro_pane_b1 = XtVaCreateManagedWidget( "macro_pane_b1",
			xmPushButtonWidgetClass,
			scriptsPane,
			RES_CONVERT( XmNlabelString, "Scripts..." ),
			NULL );
	XtAddCallback( macro_pane_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_macro_pane_b1,
		(XtPointer) NULL );



	/* Creation of macro_pane_b2 */
	macro_pane_b2 = XtVaCreateManagedWidget( "macro_pane_b2",
			xmSeparatorWidgetClass,
			scriptsPane,
			NULL );


	/* Creation of macro_pane_LandOutline */
	macro_pane_LandOutline = XtVaCreateManagedWidget( "macro_pane_LandOutline",
			xmPushButtonWidgetClass,
			scriptsPane,
			RES_CONVERT( XmNlabelString, "Land Outline" ),
			NULL );
	XtAddCallback( macro_pane_LandOutline, XmNactivateCallback,
		(XtCallbackProc) activateCB_macro_pane_LandOutline,
		(XtPointer) NULL );



	/* Creation of macro_pane_SolidLand */
	macro_pane_SolidLand = XtVaCreateManagedWidget( "macro_pane_SolidLand",
			xmPushButtonWidgetClass,
			scriptsPane,
			RES_CONVERT( XmNlabelString, "Solid Land" ),
			NULL );
	XtAddCallback( macro_pane_SolidLand, XmNactivateCallback,
		(XtCallbackProc) activateCB_macro_pane_SolidLand,
		(XtPointer) NULL );



	/* Creation of scriptsCascade */
	scriptsCascade = XtVaCreateManagedWidget( "scriptsCascade",
			xmCascadeButtonWidgetClass,
			Ferret_app_menu1,
			RES_CONVERT( XmNlabelString, "Scripts" ),
			XmNsubMenuId, scriptsPane,
			RES_CONVERT( XmNmnemonic, "M" ),
			NULL );


	/* Creation of helpPane */
	helpPane_shell = XtVaCreatePopupShell ("helpPane_shell",
			xmMenuShellWidgetClass, Ferret_app_menu1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	helpPane = XtVaCreateWidget( "helpPane",
			xmRowColumnWidgetClass,
			helpPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of AboutFerretButton */
	AboutFerretButton = XtVaCreateManagedWidget( "AboutFerretButton",
			xmPushButtonWidgetClass,
			helpPane,
			RES_CONVERT( XmNlabelString, "About Ferret..." ),
			NULL );
	XtAddCallback( AboutFerretButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_AboutFerretButton,
		(XtPointer) NULL );



	/* Creation of helpCascade */
	helpCascade = XtVaCreateManagedWidget( "helpCascade",
			xmCascadeButtonWidgetClass,
			Ferret_app_menu1,
			RES_CONVERT( XmNlabelString, "Help" ),
			RES_CONVERT( XmNmnemonic, "H" ),
			XmNsubMenuId, helpPane,
			NULL );


	/* Creation of viewPane */
	viewPane_shell = XtVaCreatePopupShell ("viewPane_shell",
			xmMenuShellWidgetClass, Ferret_app_menu1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	viewPane = XtVaCreateWidget( "viewPane",
			xmRowColumnWidgetClass,
			viewPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of InfoButton */
	InfoButton = XtVaCreateManagedWidget( "InfoButton",
			xmPushButtonWidgetClass,
			viewPane,
			RES_CONVERT( XmNlabelString, "Dataset Variables" ),
			NULL );
	XtAddCallback( InfoButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_InfoButton,
		(XtPointer) NULL );



	/* Creation of ListButton */
	ListButton = XtVaCreateManagedWidget( "ListButton",
			xmPushButtonWidgetClass,
			viewPane,
			RES_CONVERT( XmNlabelString, "Variable Values" ),
			NULL );
	XtAddCallback( ListButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_ListButton,
		(XtPointer) NULL );



	/* Creation of FerretOutputButton */
	FerretOutputButton = XtVaCreateManagedWidget( "FerretOutputButton",
			xmPushButtonWidgetClass,
			viewPane,
			RES_CONVERT( XmNlabelString, "Ferret Output..." ),
			NULL );
	XtAddCallback( FerretOutputButton, XmNactivateCallback,
		(XtCallbackProc) activateCB_FerretOutputButton,
		(XtPointer) NULL );



	/* Creation of errorlog */
	errorlog = XtVaCreateManagedWidget( "errorlog",
			xmPushButtonWidgetClass,
			viewPane,
			RES_CONVERT( XmNlabelString, "Error Log..." ),
			NULL );
	XtAddCallback( errorlog, XmNactivateCallback,
		(XtCallbackProc) activateCB_errorlog,
		(XtPointer) NULL );



	/* Creation of viewCascade */
	viewCascade = XtVaCreateManagedWidget( "viewCascade",
			xmCascadeButtonWidgetClass,
			Ferret_app_menu1,
			RES_CONVERT( XmNlabelString, "View" ),
			XmNsubMenuId, viewPane,
			RES_CONVERT( XmNmnemonic, "V" ),
			NULL );


	/* Creation of optionsPane */
	optionsPane_shell = XtVaCreatePopupShell ("optionsPane_shell",
			xmMenuShellWidgetClass, Ferret_app_menu1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionsPane = XtVaCreateWidget( "optionsPane",
			xmRowColumnWidgetClass,
			optionsPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of PlotOptions_Button */
	PlotOptions_Button = XtVaCreateManagedWidget( "PlotOptions_Button",
			xmPushButtonWidgetClass,
			optionsPane,
			RES_CONVERT( XmNlabelString, "Plot Options" ),
			NULL );
	XtAddCallback( PlotOptions_Button, XmNactivateCallback,
		(XtCallbackProc) JC_PlotOptions_CB,
		(XtPointer) NULL );



	/* Creation of Viewports_Button */
	Viewports_Button = XtVaCreateManagedWidget( "Viewports_Button",
			xmPushButtonWidgetClass,
			optionsPane,
			RES_CONVERT( XmNlabelString, "Viewports ..." ),
			NULL );
	XtAddCallback( Viewports_Button, XmNactivateCallback,
		(XtCallbackProc) JC_Viewports_CB,
		(XtPointer) NULL );



	/* Creation of SetWindows_SubPane */
	SetWindows_SubPane_shell = XtVaCreatePopupShell ("SetWindows_SubPane_shell",
			xmMenuShellWidgetClass, optionsPane,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	SetWindows_SubPane = XtVaCreateWidget( "SetWindows_SubPane",
			xmRowColumnWidgetClass,
			SetWindows_SubPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of SetWindow_1_Button */
	SetWindow_1_Button = XtVaCreateManagedWidget( "SetWindow_1_Button",
			xmPushButtonWidgetClass,
			SetWindows_SubPane,
			RES_CONVERT( XmNlabelString, "SET WINDOW 1" ),
			NULL );
	XtAddCallback( SetWindow_1_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of SetWindow_2_Button */
	SetWindow_2_Button = XtVaCreateManagedWidget( "SetWindow_2_Button",
			xmPushButtonWidgetClass,
			SetWindows_SubPane,
			RES_CONVERT( XmNlabelString, "SET WINDOW 2" ),
			NULL );
	XtAddCallback( SetWindow_2_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of SetWindow_3_Button */
	SetWindow_3_Button = XtVaCreateManagedWidget( "SetWindow_3_Button",
			xmPushButtonWidgetClass,
			SetWindows_SubPane,
			RES_CONVERT( XmNlabelString, "SET WINDOW 3" ),
			NULL );
	XtAddCallback( SetWindow_3_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of SetWindow_4_Button */
	SetWindow_4_Button = XtVaCreateManagedWidget( "SetWindow_4_Button",
			xmPushButtonWidgetClass,
			SetWindows_SubPane,
			RES_CONVERT( XmNlabelString, "SET WINDOW 4" ),
			NULL );
	XtAddCallback( SetWindow_4_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of SetWindow_5_Button */
	SetWindow_5_Button = XtVaCreateManagedWidget( "SetWindow_5_Button",
			xmPushButtonWidgetClass,
			SetWindows_SubPane,
			RES_CONVERT( XmNlabelString, "SET WINDOW 5" ),
			NULL );
	XtAddCallback( SetWindow_5_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of SetWindow_Cascade */
	SetWindow_Cascade = XtVaCreateManagedWidget( "SetWindow_Cascade",
			xmCascadeButtonWidgetClass,
			optionsPane,
			RES_CONVERT( XmNlabelString, "Set Windows" ),
			XmNsubMenuId, SetWindows_SubPane,
			NULL );


	/* Creation of CancelWindows_SubPane */
	CancelWindows_SubPane_shell = XtVaCreatePopupShell ("CancelWindows_SubPane_shell",
			xmMenuShellWidgetClass, optionsPane,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	CancelWindows_SubPane = XtVaCreateWidget( "CancelWindows_SubPane",
			xmRowColumnWidgetClass,
			CancelWindows_SubPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of CancelWindow_1_Button */
	CancelWindow_1_Button = XtVaCreateManagedWidget( "CancelWindow_1_Button",
			xmPushButtonWidgetClass,
			CancelWindows_SubPane,
			RES_CONVERT( XmNlabelString, "CANCEL WINDOW 1" ),
			NULL );
	XtAddCallback( CancelWindow_1_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of CancelWindow_2_Button */
	CancelWindow_2_Button = XtVaCreateManagedWidget( "CancelWindow_2_Button",
			xmPushButtonWidgetClass,
			CancelWindows_SubPane,
			RES_CONVERT( XmNlabelString, "CANCEL WINDOW 2" ),
			NULL );
	XtAddCallback( CancelWindow_2_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of CancelWindow_3_Button */
	CancelWindow_3_Button = XtVaCreateManagedWidget( "CancelWindow_3_Button",
			xmPushButtonWidgetClass,
			CancelWindows_SubPane,
			RES_CONVERT( XmNlabelString, "CANCEL WINDOW 3" ),
			NULL );
	XtAddCallback( CancelWindow_3_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of CancelWindow_4_Button */
	CancelWindow_4_Button = XtVaCreateManagedWidget( "CancelWindow_4_Button",
			xmPushButtonWidgetClass,
			CancelWindows_SubPane,
			RES_CONVERT( XmNlabelString, "CANCEL WINDOW 4" ),
			NULL );
	XtAddCallback( CancelWindow_4_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of CancelWindow_5_Button */
	CancelWindow_5_Button = XtVaCreateManagedWidget( "CancelWindow_5_Button",
			xmPushButtonWidgetClass,
			CancelWindows_SubPane,
			RES_CONVERT( XmNlabelString, "CANCEL WINDOW 5" ),
			NULL );
	XtAddCallback( CancelWindow_5_Button, XmNactivateCallback,
		(XtCallbackProc) JC_WindowButton_CB,
		(XtPointer) NULL );



	/* Creation of CancelWindow_Cascade */
	CancelWindow_Cascade = XtVaCreateManagedWidget( "CancelWindow_Cascade",
			xmCascadeButtonWidgetClass,
			optionsPane,
			RES_CONVERT( XmNlabelString, "Cancel Windows" ),
			XmNsubMenuId, CancelWindows_SubPane,
			NULL );


	/* Creation of Separator */
	Separator = XtVaCreateManagedWidget( "Separator",
			xmSeparatorWidgetClass,
			optionsPane,
			NULL );


	/* Creation of IncludeHours_Button */
	IncludeHours_Button = XtVaCreateManagedWidget( "IncludeHours_Button",
			xmToggleButtonWidgetClass,
			optionsPane,
			RES_CONVERT( XmNlabelString, "Time: include H:M:S" ),
			NULL );
	XtAddCallback( IncludeHours_Button, XmNvalueChangedCallback,
		(XtCallbackProc) JC_IncludeHours_CB,
		(XtPointer) NULL );



	/* Creation of ShowMap_Button */
	ShowMap_Button = XtVaCreateManagedWidget( "ShowMap_Button",
			xmToggleButtonWidgetClass,
			optionsPane,
			RES_CONVERT( XmNlabelString, "Show Map" ),
			NULL );
	XtAddCallback( ShowMap_Button, XmNvalueChangedCallback,
		(XtCallbackProc) JC_MapShowHide_CB,
		(XtPointer) NULL );



	/* Creation of optionsCascade */
	optionsCascade = XtVaCreateManagedWidget( "optionsCascade",
			xmCascadeButtonWidgetClass,
			Ferret_app_menu1,
			RES_CONVERT( XmNlabelString, "Options" ),
			RES_CONVERT( XmNmnemonic, "O" ),
			XmNsubMenuId, optionsPane,
			NULL );


	/* Creation of debugPane */
	debugPane_shell = XtVaCreatePopupShell ("debugPane_shell",
			xmMenuShellWidgetClass, Ferret_app_menu1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	debugPane = XtVaCreateWidget( "debugPane",
			xmRowColumnWidgetClass,
			debugPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of PrintAxis_SubPane */
	PrintAxis_SubPane_shell = XtVaCreatePopupShell ("PrintAxis_SubPane_shell",
			xmMenuShellWidgetClass, debugPane,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	PrintAxis_SubPane = XtVaCreateWidget( "PrintAxis_SubPane",
			xmRowColumnWidgetClass,
			PrintAxis_SubPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of X_AxisButton */
	X_AxisButton = XtVaCreateManagedWidget( "X_AxisButton",
			xmPushButtonWidgetClass,
			PrintAxis_SubPane,
			RES_CONVERT( XmNlabelString, "X Axis" ),
			NULL );
	XtAddCallback( X_AxisButton, XmNactivateCallback,
		(XtCallbackProc) JC_X_AxisButton_CB,
		(XtPointer) NULL );



	/* Creation of Y_AxisButton */
	Y_AxisButton = XtVaCreateManagedWidget( "Y_AxisButton",
			xmPushButtonWidgetClass,
			PrintAxis_SubPane,
			RES_CONVERT( XmNlabelString, "Y Axis" ),
			NULL );
	XtAddCallback( Y_AxisButton, XmNactivateCallback,
		(XtCallbackProc) JC_Y_AxisButton_CB,
		(XtPointer) NULL );



	/* Creation of Z_AxisButton */
	Z_AxisButton = XtVaCreateManagedWidget( "Z_AxisButton",
			xmPushButtonWidgetClass,
			PrintAxis_SubPane,
			RES_CONVERT( XmNlabelString, "Z Axis" ),
			NULL );
	XtAddCallback( Z_AxisButton, XmNactivateCallback,
		(XtCallbackProc) JC_Z_AxisButton_CB,
		(XtPointer) NULL );



	/* Creation of T_AxisButton */
	T_AxisButton = XtVaCreateManagedWidget( "T_AxisButton",
			xmPushButtonWidgetClass,
			PrintAxis_SubPane,
			RES_CONVERT( XmNlabelString, "T Axis" ),
			NULL );
	XtAddCallback( T_AxisButton, XmNactivateCallback,
		(XtCallbackProc) JC_T_AxisButton_CB,
		(XtPointer) NULL );



	/* Creation of PrintAxisButton */
	PrintAxisButton = XtVaCreateManagedWidget( "PrintAxisButton",
			xmCascadeButtonWidgetClass,
			debugPane,
			RES_CONVERT( XmNlabelString, "Print Axis" ),
			XmNsubMenuId, PrintAxis_SubPane,
			NULL );


	/* Creation of PrintSpan_SubPane */
	PrintSpan_SubPane_shell = XtVaCreatePopupShell ("PrintSpan_SubPane_shell",
			xmMenuShellWidgetClass, debugPane,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	PrintSpan_SubPane = XtVaCreateWidget( "PrintSpan_SubPane",
			xmRowColumnWidgetClass,
			PrintSpan_SubPane_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of X_SpanButton */
	X_SpanButton = XtVaCreateManagedWidget( "X_SpanButton",
			xmPushButtonWidgetClass,
			PrintSpan_SubPane,
			RES_CONVERT( XmNlabelString, "X Span" ),
			NULL );
	XtAddCallback( X_SpanButton, XmNactivateCallback,
		(XtCallbackProc) JC_X_SpanButton_CB,
		(XtPointer) NULL );



	/* Creation of Y_SpanButton */
	Y_SpanButton = XtVaCreateManagedWidget( "Y_SpanButton",
			xmPushButtonWidgetClass,
			PrintSpan_SubPane,
			RES_CONVERT( XmNlabelString, "Y Span" ),
			NULL );
	XtAddCallback( Y_SpanButton, XmNactivateCallback,
		(XtCallbackProc) JC_Y_SpanButton_CB,
		(XtPointer) NULL );



	/* Creation of Z_SpanButton */
	Z_SpanButton = XtVaCreateManagedWidget( "Z_SpanButton",
			xmPushButtonWidgetClass,
			PrintSpan_SubPane,
			RES_CONVERT( XmNlabelString, "Z Span" ),
			NULL );
	XtAddCallback( Z_SpanButton, XmNactivateCallback,
		(XtCallbackProc) JC_Z_SpanButton_CB,
		(XtPointer) NULL );



	/* Creation of T_SpanButton */
	T_SpanButton = XtVaCreateManagedWidget( "T_SpanButton",
			xmPushButtonWidgetClass,
			PrintSpan_SubPane,
			RES_CONVERT( XmNlabelString, "T Span" ),
			NULL );
	XtAddCallback( T_SpanButton, XmNactivateCallback,
		(XtCallbackProc) JC_T_SpanButton_CB,
		(XtPointer) NULL );



	/* Creation of PrintSpanButton */
	PrintSpanButton = XtVaCreateManagedWidget( "PrintSpanButton",
			xmCascadeButtonWidgetClass,
			debugPane,
			RES_CONVERT( XmNlabelString, "Print Span" ),
			XmNsubMenuId, PrintSpan_SubPane,
			NULL );


	/* Creation of debugCascade */
	debugCascade = XtVaCreateManagedWidget( "debugCascade",
			xmCascadeButtonWidgetClass,
			Ferret_app_menu1,
			RES_CONVERT( XmNlabelString, "Debug" ),
			XmNsubMenuId, debugPane,
			NULL );


	/* Creation of frame_Data */
	frame_Data = XtVaCreateManagedWidget( "frame_Data",
			xmFrameWidgetClass,
			form1,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, Ferret_app_menu1,
			XmNtopOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of label_Data */
	label_Data = XtVaCreateManagedWidget( "label_Data",
			xmLabelWidgetClass,
			frame_Data,
			RES_CONVERT( XmNlabelString, "Data" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_Data */
	form_Data = XtVaCreateManagedWidget( "form_Data",
			xmFormWidgetClass,
			frame_Data,
			NULL );


	/* Creation of rowColumn_Select */
	rowColumn_Select = XtVaCreateManagedWidget( "rowColumn_Select",
			xmRowColumnWidgetClass,
			form_Data,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of rowColumn_dummy */
	rowColumn_dummy = XtVaCreateManagedWidget( "rowColumn_dummy",
			xmRowColumnWidgetClass,
			form_Data,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of pushButton_dummy */
	pushButton_dummy = XtVaCreateManagedWidget( "pushButton_dummy",
			xmPushButtonWidgetClass,
			rowColumn_dummy,
			RES_CONVERT( XmNlabelString, "Select" ),
			NULL );


	/* Creation of rowColumn_Data */
	rowColumn_Data = XtVaCreateManagedWidget( "rowColumn_Data",
			xmRowColumnWidgetClass,
			form_Data,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNleftOffset, 30,
			XmNleftWidget, rowColumn_dummy,
			NULL );


	/* Creation of pushButton_Clone */
	pushButton_Clone = XtVaCreateManagedWidget( "pushButton_Clone",
			xmPushButtonWidgetClass,
			rowColumn_Data,
			RES_CONVERT( XmNlabelString, "Clone" ),
			NULL );
	XtAddCallback( pushButton_Clone, XmNactivateCallback,
		(XtCallbackProc) JC_CloneButton_CB,
		(XtPointer) NULL );



	/* Creation of textField_Variable */
	textField_Variable = XtVaCreateManagedWidget( "textField_Variable",
			xmTextFieldWidgetClass,
			rowColumn_Data,
			XmNcursorPositionVisible, FALSE,
			XmNeditable, FALSE,
			NULL );
	XtAddCallback( textField_Variable, XmNactivateCallback,
		(XtCallbackProc) JC_VariableTextField_Verify_CB,
		(XtPointer) NULL );
	XtAddCallback( textField_Variable, XmNlosingFocusCallback,
		(XtCallbackProc) JC_VariableTextField_Verify_CB,
		(XtPointer) NULL );



	/* Creation of textField_Dataset */
	textField_Dataset = XtVaCreateManagedWidget( "textField_Dataset",
			xmTextFieldWidgetClass,
			rowColumn_Data,
			XmNcursorPositionVisible, FALSE,
			XmNeditable, FALSE,
			NULL );


	/* Creation of label1 */
	label1 = XtVaCreateManagedWidget( "label1",
			xmLabelWidgetClass,
			rowColumn_Data,
			RES_CONVERT( XmNlabelString, "of" ),
			NULL );


	/* Creation of label_DataFrameStatus */
	label_DataFrameStatus = XtVaCreateManagedWidget( "label_DataFrameStatus",
			xmLabelWidgetClass,
			rowColumn_Data,
			XmNx, 280,
			XmNy, 10,
			XmNwidth, 50,
			XmNheight, 10,
			NULL );


	/* Creation of frame_context */
	frame_context = XtVaCreateManagedWidget( "frame_context",
			xmFrameWidgetClass,
			form1,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNtopOffset, 5,
			XmNtopWidget, frame_Data,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of label65 */
	label65 = XtVaCreateManagedWidget( "label65",
			xmLabelWidgetClass,
			frame_context,
			RES_CONVERT( XmNlabelString, "Context" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_Context */
	form_Context = XtVaCreateManagedWidget( "form_Context",
			xmFormWidgetClass,
			frame_context,
			NULL );


	/* Creation of form_Transforms */
	form_Transforms = XtVaCreateManagedWidget( "form_Transforms",
			xmFormWidgetClass,
			form_Context,
			XmNleftAttachment, XmATTACH_NONE,
			XmNrightAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_FORM,
			XmNwidth, 200,
			XmNtopOffset, 0,
			NULL );


	/* Creation of label_Transform */
	label_Transform = XtVaCreateManagedWidget( "label_Transform",
			xmLabelWidgetClass,
			form_Transforms,
			RES_CONVERT( XmNlabelString, "Transform" ),
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 30,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 40,
			NULL );


	/* Creation of label_Argument */
	label_Argument = XtVaCreateManagedWidget( "label_Argument",
			xmLabelWidgetClass,
			form_Transforms,
			RES_CONVERT( XmNlabelString, "Arg" ),
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNleftOffset, 20,
			XmNleftWidget, label_Transform,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 40,
			NULL );


	/* Creation of rowColumn_TRANS_X */
	rowColumn_TRANS_X = XtVaCreateManagedWidget( "rowColumn_TRANS_X",
			xmRowColumnWidgetClass,
			form_Transforms,
			XmNwidth, 100,
			XmNheight, 80,
			XmNx, 30,
			XmNy, 70,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 63,
			NULL );


	/* Creation of optionMenu_5 */
	optionMenu_5_shell = XtVaCreatePopupShell ("optionMenu_5_shell",
			xmMenuShellWidgetClass, rowColumn_TRANS_X,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_5 = XtVaCreateWidget( "optionMenu_5",
			xmRowColumnWidgetClass,
			optionMenu_5_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 497,
			XmNy, 0,
			XmNnumColumns, 2,
			XmNpacking, XmPACK_COLUMN,
			XmNtearOffModel, XmTEAR_OFF_DISABLED,
			XmNorientation, XmVERTICAL,
			XmNheight, 20,
			XmNwidth, 100,
			XmNmappedWhenManaged, TRUE,
			NULL );


	/* Creation of optionMenu_p_b8 */
	optionMenu_p_b8 = XtVaCreateManagedWidget( "optionMenu_p_b8",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "None" ),
			XmNx, 497,
			XmNy, 2,
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			NULL );
	XtAddCallback( optionMenu_p_b8, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p_b8,
		(XtPointer) "x_non" );



	/* Creation of optionMenu_2_b91 */
	optionMenu_2_b91 = XtVaCreateManagedWidget( "optionMenu_2_b91",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "ave" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 23,
			NULL );
	XtAddCallback( optionMenu_2_b91, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b91,
		(XtPointer) "x_ave" );



	/* Creation of optionMenu_2_b92 */
	optionMenu_2_b92 = XtVaCreateManagedWidget( "optionMenu_2_b92",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "var" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 44,
			NULL );
	XtAddCallback( optionMenu_2_b92, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b92,
		(XtPointer) "x_var" );



	/* Creation of optionMenu_2_b93 */
	optionMenu_2_b93 = XtVaCreateManagedWidget( "optionMenu_2_b93",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "sum" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 65,
			NULL );
	XtAddCallback( optionMenu_2_b93, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b93,
		(XtPointer) "x_sum" );



	/* Creation of optionMenu_2_b94 */
	optionMenu_2_b94 = XtVaCreateManagedWidget( "optionMenu_2_b94",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "rsu" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 86,
			NULL );
	XtAddCallback( optionMenu_2_b94, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b94,
		(XtPointer) "x_rsu" );



	/* Creation of optionMenu_2_b95 */
	optionMenu_2_b95 = XtVaCreateManagedWidget( "optionMenu_2_b95",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "shf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 107,
			NULL );
	XtAddCallback( optionMenu_2_b95, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b95,
		(XtPointer) "x_shf" );



	/* Creation of optionMenu_2_b96 */
	optionMenu_2_b96 = XtVaCreateManagedWidget( "optionMenu_2_b96",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, " Min" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 128,
			NULL );
	XtAddCallback( optionMenu_2_b96, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b96,
		(XtPointer) "x_min" );



	/* Creation of optionMenu_2_b97 */
	optionMenu_2_b97 = XtVaCreateManagedWidget( "optionMenu_2_b97",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, " Max" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 149,
			NULL );
	XtAddCallback( optionMenu_2_b97, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b97,
		(XtPointer) "x_max" );



	/* Creation of optionMenu_2_b98 */
	optionMenu_2_b98 = XtVaCreateManagedWidget( "optionMenu_2_b98",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "ddc" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 170,
			NULL );
	XtAddCallback( optionMenu_2_b98, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b98,
		(XtPointer) "x_ddc" );



	/* Creation of optionMenu_2_b99 */
	optionMenu_2_b99 = XtVaCreateManagedWidget( "optionMenu_2_b99",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "ddf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 191,
			NULL );
	XtAddCallback( optionMenu_2_b99, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b99,
		(XtPointer) "x_ddf" );



	/* Creation of optionMenu_2_b100 */
	optionMenu_2_b100 = XtVaCreateManagedWidget( "optionMenu_2_b100",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "ddb" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 212,
			NULL );
	XtAddCallback( optionMenu_2_b100, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b100,
		(XtPointer) "x_ddb" );



	/* Creation of optionMenu_2_b101 */
	optionMenu_2_b101 = XtVaCreateManagedWidget( "optionMenu_2_b101",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "din" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 233,
			NULL );
	XtAddCallback( optionMenu_2_b101, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b101,
		(XtPointer) "x_din" );



	/* Creation of optionMenu_2_b102 */
	optionMenu_2_b102 = XtVaCreateManagedWidget( "optionMenu_2_b102",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "iin" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 254,
			NULL );
	XtAddCallback( optionMenu_2_b102, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b102,
		(XtPointer) "x_iin" );



	/* Creation of optionMenu_2_b103 */
	optionMenu_2_b103 = XtVaCreateManagedWidget( "optionMenu_2_b103",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "sbx" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 2,
			NULL );
	XtAddCallback( optionMenu_2_b103, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b103,
		(XtPointer) "x_sbx" );



	/* Creation of optionMenu_2_b104 */
	optionMenu_2_b104 = XtVaCreateManagedWidget( "optionMenu_2_b104",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "sbn" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 23,
			NULL );
	XtAddCallback( optionMenu_2_b104, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b104,
		(XtPointer) "x_sbn" );



	/* Creation of optionMenu_2_b105 */
	optionMenu_2_b105 = XtVaCreateManagedWidget( "optionMenu_2_b105",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "sbh" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 44,
			NULL );
	XtAddCallback( optionMenu_2_b105, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b105,
		(XtPointer) "x_shn" );



	/* Creation of optionMenu_2_b106 */
	optionMenu_2_b106 = XtVaCreateManagedWidget( "optionMenu_2_b106",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "sbw" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 65,
			NULL );
	XtAddCallback( optionMenu_2_b106, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b106,
		(XtPointer) "x_swl" );



	/* Creation of optionMenu_2_b107 */
	optionMenu_2_b107 = XtVaCreateManagedWidget( "optionMenu_2_b107",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "spz" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 86,
			NULL );
	XtAddCallback( optionMenu_2_b107, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b107,
		(XtPointer) "x_spz" );



	/* Creation of optionMenu_2_b108 */
	optionMenu_2_b108 = XtVaCreateManagedWidget( "optionMenu_2_b108",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "fav" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 107,
			NULL );
	XtAddCallback( optionMenu_2_b108, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b108,
		(XtPointer) "x_fav" );



	/* Creation of optionMenu_2_b109 */
	optionMenu_2_b109 = XtVaCreateManagedWidget( "optionMenu_2_b109",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "fln" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 128,
			NULL );
	XtAddCallback( optionMenu_2_b109, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b109,
		(XtPointer) "x_fln" );



	/* Creation of optionMenu_2_b110 */
	optionMenu_2_b110 = XtVaCreateManagedWidget( "optionMenu_2_b110",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, "fnr" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 149,
			NULL );
	XtAddCallback( optionMenu_2_b110, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b110,
		(XtPointer) "x_fnr" );



	/* Creation of optionMenu_2_b111 */
	optionMenu_2_b111 = XtVaCreateManagedWidget( "optionMenu_2_b111",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, " # Bad " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 170,
			NULL );
	XtAddCallback( optionMenu_2_b111, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b111,
		(XtPointer) "x_nbd" );



	/* Creation of optionMenu_2_b112 */
	optionMenu_2_b112 = XtVaCreateManagedWidget( "optionMenu_2_b112",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, " # Good " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 191,
			NULL );
	XtAddCallback( optionMenu_2_b112, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b112,
		(XtPointer) "x_ngd" );



	/* Creation of optionMenu_2_b113 */
	optionMenu_2_b113 = XtVaCreateManagedWidget( "optionMenu_2_b113",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, " Locate" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 212,
			NULL );
	XtAddCallback( optionMenu_2_b113, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b113,
		(XtPointer) "x_loc" );



	/* Creation of optionMenu_2_b114 */
	optionMenu_2_b114 = XtVaCreateManagedWidget( "optionMenu_2_b114",
			xmPushButtonWidgetClass,
			optionMenu_5,
			RES_CONVERT( XmNlabelString, " WEQ" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 233,
			NULL );
	XtAddCallback( optionMenu_2_b114, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b114,
		(XtPointer) "x_weq" );



	/* Creation of optionMenu_X_TRANS */
	optionMenu_X_TRANS = XtVaCreateManagedWidget( "optionMenu_X_TRANS",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_X,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_5,
			RES_CONVERT( XmNlabelString, " " ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNmappedWhenManaged, TRUE,
			XmNwidth, 104,
			XmNy, 28,
			NULL );


	/* Creation of rowColumn_SubTrans */
	rowColumn_SubTrans = XtVaCreateManagedWidget( "rowColumn_SubTrans",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_X,
			XmNwidth, 50,
			XmNheight, 30,
			XmNx, 80,
			XmNy, 60,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of textField_X_ARG */
	textField_X_ARG = XtVaCreateManagedWidget( "textField_X_ARG",
			xmTextFieldWidgetClass,
			rowColumn_SubTrans,
			XmNcolumns, 5,
			NULL );
	XtAddCallback( textField_X_ARG, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField_X_ARG,
		(XtPointer) NULL );
	XtAddCallback( textField_X_ARG, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_X_ARG,
		"x" );
	XtAddCallback( textField_X_ARG, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_X_ARG,
		"x" );



	/* Creation of rowColumn_TRANS_Y */
	rowColumn_TRANS_Y = XtVaCreateManagedWidget( "rowColumn_TRANS_Y",
			xmRowColumnWidgetClass,
			form_Transforms,
			XmNwidth, 100,
			XmNheight, 80,
			XmNx, 30,
			XmNy, 70,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 117,
			NULL );


	/* Creation of optionMenu_1 */
	optionMenu_1_shell = XtVaCreatePopupShell ("optionMenu_1_shell",
			xmMenuShellWidgetClass, rowColumn_TRANS_Y,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_1 = XtVaCreateWidget( "optionMenu_1",
			xmRowColumnWidgetClass,
			optionMenu_1_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 497,
			XmNy, 0,
			XmNnumColumns, 2,
			XmNpacking, XmPACK_COLUMN,
			XmNtearOffModel, XmTEAR_OFF_DISABLED,
			XmNorientation, XmVERTICAL,
			XmNheight, 20,
			XmNwidth, 100,
			XmNmappedWhenManaged, TRUE,
			NULL );


	/* Creation of optionMenu_p_b9 */
	optionMenu_p_b9 = XtVaCreateManagedWidget( "optionMenu_p_b9",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, " None" ),
			XmNx, 497,
			XmNy, 2,
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			NULL );
	XtAddCallback( optionMenu_p_b9, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p_b9,
		(XtPointer) "y_non" );



	/* Creation of optionMenu_2_b1 */
	optionMenu_2_b1 = XtVaCreateManagedWidget( "optionMenu_2_b1",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "ave" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 23,
			NULL );
	XtAddCallback( optionMenu_2_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b1,
		(XtPointer) "y_ave" );



	/* Creation of optionMenu_2_b6 */
	optionMenu_2_b6 = XtVaCreateManagedWidget( "optionMenu_2_b6",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "var" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 44,
			NULL );
	XtAddCallback( optionMenu_2_b6, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b6,
		(XtPointer) "y_var" );



	/* Creation of optionMenu_2_b25 */
	optionMenu_2_b25 = XtVaCreateManagedWidget( "optionMenu_2_b25",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "sum" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 65,
			NULL );
	XtAddCallback( optionMenu_2_b25, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b25,
		(XtPointer) "y_sum" );



	/* Creation of optionMenu_2_b26 */
	optionMenu_2_b26 = XtVaCreateManagedWidget( "optionMenu_2_b26",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "rsu" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 86,
			NULL );
	XtAddCallback( optionMenu_2_b26, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b26,
		(XtPointer) "y_rsu" );



	/* Creation of optionMenu_2_b27 */
	optionMenu_2_b27 = XtVaCreateManagedWidget( "optionMenu_2_b27",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "shf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 107,
			NULL );
	XtAddCallback( optionMenu_2_b27, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b27,
		(XtPointer) "y_shf" );



	/* Creation of optionMenu_2_b28 */
	optionMenu_2_b28 = XtVaCreateManagedWidget( "optionMenu_2_b28",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, " Min" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 128,
			NULL );
	XtAddCallback( optionMenu_2_b28, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b28,
		(XtPointer) "y_min" );



	/* Creation of optionMenu_2_b29 */
	optionMenu_2_b29 = XtVaCreateManagedWidget( "optionMenu_2_b29",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, " Max" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 149,
			NULL );
	XtAddCallback( optionMenu_2_b29, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b29,
		(XtPointer) "y_max" );



	/* Creation of optionMenu_2_b30 */
	optionMenu_2_b30 = XtVaCreateManagedWidget( "optionMenu_2_b30",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "ddc" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 170,
			NULL );
	XtAddCallback( optionMenu_2_b30, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b30,
		(XtPointer) "y_ddc" );



	/* Creation of optionMenu_2_b31 */
	optionMenu_2_b31 = XtVaCreateManagedWidget( "optionMenu_2_b31",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "ddf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 191,
			NULL );
	XtAddCallback( optionMenu_2_b31, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b31,
		(XtPointer) "y_ddf" );



	/* Creation of optionMenu_2_b32 */
	optionMenu_2_b32 = XtVaCreateManagedWidget( "optionMenu_2_b32",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "ddb" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 212,
			NULL );
	XtAddCallback( optionMenu_2_b32, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b32,
		(XtPointer) "y_ddb" );



	/* Creation of optionMenu_2_b33 */
	optionMenu_2_b33 = XtVaCreateManagedWidget( "optionMenu_2_b33",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "din" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 233,
			NULL );
	XtAddCallback( optionMenu_2_b33, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b33,
		(XtPointer) "y_din" );



	/* Creation of optionMenu_2_b34 */
	optionMenu_2_b34 = XtVaCreateManagedWidget( "optionMenu_2_b34",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "iin" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 254,
			NULL );
	XtAddCallback( optionMenu_2_b34, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b34,
		(XtPointer) "y_iin" );



	/* Creation of optionMenu_2_b35 */
	optionMenu_2_b35 = XtVaCreateManagedWidget( "optionMenu_2_b35",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "sbx" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 2,
			NULL );
	XtAddCallback( optionMenu_2_b35, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b35,
		(XtPointer) "y_sbx" );



	/* Creation of optionMenu_2_b36 */
	optionMenu_2_b36 = XtVaCreateManagedWidget( "optionMenu_2_b36",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "sbn" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 23,
			NULL );
	XtAddCallback( optionMenu_2_b36, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b36,
		(XtPointer) "y_sbn" );



	/* Creation of optionMenu_2_b37 */
	optionMenu_2_b37 = XtVaCreateManagedWidget( "optionMenu_2_b37",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "sbh" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 44,
			NULL );
	XtAddCallback( optionMenu_2_b37, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b37,
		(XtPointer) "y_shn" );



	/* Creation of optionMenu_2_b38 */
	optionMenu_2_b38 = XtVaCreateManagedWidget( "optionMenu_2_b38",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "sbw" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 65,
			NULL );
	XtAddCallback( optionMenu_2_b38, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b38,
		(XtPointer) "y_swl" );



	/* Creation of optionMenu_2_b39 */
	optionMenu_2_b39 = XtVaCreateManagedWidget( "optionMenu_2_b39",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "spz" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 86,
			NULL );
	XtAddCallback( optionMenu_2_b39, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b39,
		(XtPointer) "y_spz" );



	/* Creation of optionMenu_2_b40 */
	optionMenu_2_b40 = XtVaCreateManagedWidget( "optionMenu_2_b40",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "fav" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 107,
			NULL );
	XtAddCallback( optionMenu_2_b40, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b40,
		(XtPointer) "y_fav" );



	/* Creation of optionMenu_2_b41 */
	optionMenu_2_b41 = XtVaCreateManagedWidget( "optionMenu_2_b41",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "fln" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 128,
			NULL );
	XtAddCallback( optionMenu_2_b41, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b41,
		(XtPointer) "y_fln" );



	/* Creation of optionMenu_1_b1 */
	optionMenu_1_b1 = XtVaCreateManagedWidget( "optionMenu_1_b1",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, "fnr" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 149,
			NULL );
	XtAddCallback( optionMenu_1_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_1_b1,
		(XtPointer) "y_fnr" );



	/* Creation of optionMenu_2_b42 */
	optionMenu_2_b42 = XtVaCreateManagedWidget( "optionMenu_2_b42",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, " # Bad  " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 170,
			NULL );
	XtAddCallback( optionMenu_2_b42, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b42,
		(XtPointer) "y_nbd" );



	/* Creation of optionMenu_2_b43 */
	optionMenu_2_b43 = XtVaCreateManagedWidget( "optionMenu_2_b43",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, " # Good " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 191,
			NULL );
	XtAddCallback( optionMenu_2_b43, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b43,
		(XtPointer) "y_ngd" );



	/* Creation of optionMenu_2_b44 */
	optionMenu_2_b44 = XtVaCreateManagedWidget( "optionMenu_2_b44",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, " Locate" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 212,
			NULL );
	XtAddCallback( optionMenu_2_b44, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b44,
		(XtPointer) "y_loc" );



	/* Creation of optionMenu_1_b2 */
	optionMenu_1_b2 = XtVaCreateManagedWidget( "optionMenu_1_b2",
			xmPushButtonWidgetClass,
			optionMenu_1,
			RES_CONVERT( XmNlabelString, " WEQ" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 233,
			NULL );
	XtAddCallback( optionMenu_1_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_1_b2,
		(XtPointer) "y_weq" );



	/* Creation of optionMenu_Y_TRANS */
	optionMenu_Y_TRANS = XtVaCreateManagedWidget( "optionMenu_Y_TRANS",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_Y,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_1,
			RES_CONVERT( XmNlabelString, " " ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNmappedWhenManaged, TRUE,
			XmNy, 82,
			XmNheight, 35,
			XmNx, 0,
			XmNwidth, 104,
			NULL );


	/* Creation of rowColumn_SubTrans1 */
	rowColumn_SubTrans1 = XtVaCreateManagedWidget( "rowColumn_SubTrans1",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_Y,
			XmNwidth, 50,
			XmNheight, 30,
			XmNx, 80,
			XmNy, 60,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of textField_Y_ARG */
	textField_Y_ARG = XtVaCreateManagedWidget( "textField_Y_ARG",
			xmTextFieldWidgetClass,
			rowColumn_SubTrans1,
			XmNsensitive, TRUE,
			XmNmappedWhenManaged, TRUE,
			XmNheight, 32,
			XmNcolumns, 5,
			NULL );
	XtAddCallback( textField_Y_ARG, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField_Y_ARG,
		(XtPointer) NULL );
	XtAddCallback( textField_Y_ARG, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_Y_ARG,
		"y" );
	XtAddCallback( textField_Y_ARG, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_Y_ARG,
		"y" );



	/* Creation of rowColumn_TRANS_Z */
	rowColumn_TRANS_Z = XtVaCreateManagedWidget( "rowColumn_TRANS_Z",
			xmRowColumnWidgetClass,
			form_Transforms,
			XmNwidth, 100,
			XmNheight, 80,
			XmNx, 30,
			XmNy, 70,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 171,
			NULL );


	/* Creation of optionMenu_2 */
	optionMenu_2_shell = XtVaCreatePopupShell ("optionMenu_2_shell",
			xmMenuShellWidgetClass, rowColumn_TRANS_Z,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_2 = XtVaCreateWidget( "optionMenu_2",
			xmRowColumnWidgetClass,
			optionMenu_2_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 497,
			XmNy, 0,
			XmNnumColumns, 2,
			XmNpacking, XmPACK_COLUMN,
			XmNtearOffModel, XmTEAR_OFF_DISABLED,
			XmNorientation, XmVERTICAL,
			XmNheight, 20,
			XmNwidth, 100,
			XmNmappedWhenManaged, TRUE,
			NULL );


	/* Creation of optionMenu_p_b2 */
	optionMenu_p_b2 = XtVaCreateManagedWidget( "optionMenu_p_b2",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, " None" ),
			XmNx, 497,
			XmNy, 2,
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			NULL );
	XtAddCallback( optionMenu_p_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p_b2,
		(XtPointer) "z_non" );



	/* Creation of optionMenu_2_b2 */
	optionMenu_2_b2 = XtVaCreateManagedWidget( "optionMenu_2_b2",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "ave" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 23,
			NULL );
	XtAddCallback( optionMenu_2_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b2,
		(XtPointer) "z_ave" );



	/* Creation of optionMenu_2_b3 */
	optionMenu_2_b3 = XtVaCreateManagedWidget( "optionMenu_2_b3",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "var" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 44,
			NULL );
	XtAddCallback( optionMenu_2_b3, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b3,
		(XtPointer) "z_var" );



	/* Creation of optionMenu_2_b4 */
	optionMenu_2_b4 = XtVaCreateManagedWidget( "optionMenu_2_b4",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "sum" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 65,
			NULL );
	XtAddCallback( optionMenu_2_b4, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b4,
		(XtPointer) "z_sum" );



	/* Creation of optionMenu_2_b5 */
	optionMenu_2_b5 = XtVaCreateManagedWidget( "optionMenu_2_b5",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "rsu" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 86,
			NULL );
	XtAddCallback( optionMenu_2_b5, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b5,
		(XtPointer) "z_rsu" );



	/* Creation of optionMenu_2_b7 */
	optionMenu_2_b7 = XtVaCreateManagedWidget( "optionMenu_2_b7",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "shf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 107,
			NULL );
	XtAddCallback( optionMenu_2_b7, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b7,
		(XtPointer) "z_shf" );



	/* Creation of optionMenu_2_b8 */
	optionMenu_2_b8 = XtVaCreateManagedWidget( "optionMenu_2_b8",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, " Min" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 128,
			NULL );
	XtAddCallback( optionMenu_2_b8, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b8,
		(XtPointer) "z_min" );



	/* Creation of optionMenu_2_b9 */
	optionMenu_2_b9 = XtVaCreateManagedWidget( "optionMenu_2_b9",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, " Max" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 149,
			NULL );
	XtAddCallback( optionMenu_2_b9, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b9,
		(XtPointer) "z_max" );



	/* Creation of optionMenu_2_b10 */
	optionMenu_2_b10 = XtVaCreateManagedWidget( "optionMenu_2_b10",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "ddc" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 170,
			NULL );
	XtAddCallback( optionMenu_2_b10, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b10,
		(XtPointer) "z_ddc" );



	/* Creation of optionMenu_2_b11 */
	optionMenu_2_b11 = XtVaCreateManagedWidget( "optionMenu_2_b11",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "ddf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 191,
			NULL );
	XtAddCallback( optionMenu_2_b11, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b11,
		(XtPointer) "z_ddf" );



	/* Creation of optionMenu_2_b12 */
	optionMenu_2_b12 = XtVaCreateManagedWidget( "optionMenu_2_b12",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "ddb" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 212,
			NULL );
	XtAddCallback( optionMenu_2_b12, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b12,
		(XtPointer) "z_ddb" );



	/* Creation of optionMenu_2_b13 */
	optionMenu_2_b13 = XtVaCreateManagedWidget( "optionMenu_2_b13",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "din" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 233,
			NULL );
	XtAddCallback( optionMenu_2_b13, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b13,
		(XtPointer) "z_din" );



	/* Creation of optionMenu_2_b14 */
	optionMenu_2_b14 = XtVaCreateManagedWidget( "optionMenu_2_b14",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "iin" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 254,
			NULL );
	XtAddCallback( optionMenu_2_b14, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b14,
		(XtPointer) "z_iin" );



	/* Creation of optionMenu_2_b15 */
	optionMenu_2_b15 = XtVaCreateManagedWidget( "optionMenu_2_b15",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "sbx" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 2,
			NULL );
	XtAddCallback( optionMenu_2_b15, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b15,
		(XtPointer) "z_sbx" );



	/* Creation of optionMenu_2_b16 */
	optionMenu_2_b16 = XtVaCreateManagedWidget( "optionMenu_2_b16",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "sbn" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 23,
			NULL );
	XtAddCallback( optionMenu_2_b16, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b16,
		(XtPointer) "z_sbn" );



	/* Creation of optionMenu_2_b17 */
	optionMenu_2_b17 = XtVaCreateManagedWidget( "optionMenu_2_b17",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "sbh" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 44,
			NULL );
	XtAddCallback( optionMenu_2_b17, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b17,
		(XtPointer) "z_shn" );



	/* Creation of optionMenu_2_b18 */
	optionMenu_2_b18 = XtVaCreateManagedWidget( "optionMenu_2_b18",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "sbw" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 65,
			NULL );
	XtAddCallback( optionMenu_2_b18, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b18,
		(XtPointer) "z_swl" );



	/* Creation of optionMenu_2_b19 */
	optionMenu_2_b19 = XtVaCreateManagedWidget( "optionMenu_2_b19",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "sbz" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 86,
			NULL );
	XtAddCallback( optionMenu_2_b19, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b19,
		(XtPointer) "z_spz" );



	/* Creation of optionMenu_2_b20 */
	optionMenu_2_b20 = XtVaCreateManagedWidget( "optionMenu_2_b20",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "fav" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 107,
			NULL );
	XtAddCallback( optionMenu_2_b20, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b20,
		(XtPointer) "z_fav" );



	/* Creation of optionMenu_2_b21 */
	optionMenu_2_b21 = XtVaCreateManagedWidget( "optionMenu_2_b21",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "fln" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 128,
			NULL );
	XtAddCallback( optionMenu_2_b21, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b21,
		(XtPointer) "z_fln" );



	/* Creation of optionMenu_3_b1 */
	optionMenu_3_b1 = XtVaCreateManagedWidget( "optionMenu_3_b1",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, "fnr" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 149,
			NULL );
	XtAddCallback( optionMenu_3_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_3_b1,
		(XtPointer) "z_fnr" );



	/* Creation of optionMenu_2_b22 */
	optionMenu_2_b22 = XtVaCreateManagedWidget( "optionMenu_2_b22",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, " # Bad  " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 170,
			NULL );
	XtAddCallback( optionMenu_2_b22, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b22,
		(XtPointer) "z_nbd" );



	/* Creation of optionMenu_2_b23 */
	optionMenu_2_b23 = XtVaCreateManagedWidget( "optionMenu_2_b23",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, " # Good " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 191,
			NULL );
	XtAddCallback( optionMenu_2_b23, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b23,
		(XtPointer) "z_ngd" );



	/* Creation of optionMenu_2_b24 */
	optionMenu_2_b24 = XtVaCreateManagedWidget( "optionMenu_2_b24",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, " Locate" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 212,
			NULL );
	XtAddCallback( optionMenu_2_b24, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b24,
		(XtPointer) "z_loc" );



	/* Creation of optionMenu_3_b2 */
	optionMenu_3_b2 = XtVaCreateManagedWidget( "optionMenu_3_b2",
			xmPushButtonWidgetClass,
			optionMenu_2,
			RES_CONVERT( XmNlabelString, " WEQ" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 233,
			NULL );
	XtAddCallback( optionMenu_3_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_3_b2,
		(XtPointer) "z_weq" );



	/* Creation of optionMenu_Z_TRANS */
	optionMenu_Z_TRANS = XtVaCreateManagedWidget( "optionMenu_Z_TRANS",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_Z,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_2,
			RES_CONVERT( XmNlabelString, " " ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNmappedWhenManaged, TRUE,
			XmNy, 138,
			XmNheight, 35,
			XmNx, 0,
			XmNwidth, 104,
			NULL );


	/* Creation of rowColumn_SubTrans2 */
	rowColumn_SubTrans2 = XtVaCreateManagedWidget( "rowColumn_SubTrans2",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_Z,
			XmNwidth, 50,
			XmNheight, 30,
			XmNx, 80,
			XmNy, 60,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of textField_Z_ARG */
	textField_Z_ARG = XtVaCreateManagedWidget( "textField_Z_ARG",
			xmTextFieldWidgetClass,
			rowColumn_SubTrans2,
			XmNsensitive, TRUE,
			XmNmappedWhenManaged, TRUE,
			XmNcolumns, 5,
			NULL );
	XtAddCallback( textField_Z_ARG, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField_Z_ARG,
		(XtPointer) NULL );
	XtAddCallback( textField_Z_ARG, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_Z_ARG,
		"z" );
	XtAddCallback( textField_Z_ARG, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_Z_ARG,
		"z" );



	/* Creation of rowColumn_TRANS_T */
	rowColumn_TRANS_T = XtVaCreateManagedWidget( "rowColumn_TRANS_T",
			xmRowColumnWidgetClass,
			form_Transforms,
			XmNwidth, 100,
			XmNheight, 80,
			XmNx, 30,
			XmNy, 70,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 225,
			NULL );


	/* Creation of optionMenu_3 */
	optionMenu_3_shell = XtVaCreatePopupShell ("optionMenu_3_shell",
			xmMenuShellWidgetClass, rowColumn_TRANS_T,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_3 = XtVaCreateWidget( "optionMenu_3",
			xmRowColumnWidgetClass,
			optionMenu_3_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 497,
			XmNy, 0,
			XmNnumColumns, 2,
			XmNpacking, XmPACK_COLUMN,
			XmNtearOffModel, XmTEAR_OFF_DISABLED,
			XmNorientation, XmVERTICAL,
			XmNheight, 20,
			XmNwidth, 100,
			XmNmappedWhenManaged, TRUE,
			NULL );


	/* Creation of optionMenu_p_b12 */
	optionMenu_p_b12 = XtVaCreateManagedWidget( "optionMenu_p_b12",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, " None" ),
			XmNx, 497,
			XmNy, 12,
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			NULL );
	XtAddCallback( optionMenu_p_b12, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p_b12,
		(XtPointer) "t_non" );



	/* Creation of optionMenu_2_b45 */
	optionMenu_2_b45 = XtVaCreateManagedWidget( "optionMenu_2_b45",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "ave" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 33,
			NULL );
	XtAddCallback( optionMenu_2_b45, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b45,
		(XtPointer) "t_ave" );



	/* Creation of optionMenu_2_b46 */
	optionMenu_2_b46 = XtVaCreateManagedWidget( "optionMenu_2_b46",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "var" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 54,
			NULL );
	XtAddCallback( optionMenu_2_b46, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b46,
		(XtPointer) "t_var" );



	/* Creation of optionMenu_2_b47 */
	optionMenu_2_b47 = XtVaCreateManagedWidget( "optionMenu_2_b47",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "sum" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 75,
			NULL );
	XtAddCallback( optionMenu_2_b47, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b47,
		(XtPointer) "t_sum" );



	/* Creation of optionMenu_2_b48 */
	optionMenu_2_b48 = XtVaCreateManagedWidget( "optionMenu_2_b48",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "rsu" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNwidth, 45,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNx, 497,
			XmNy, 96,
			NULL );
	XtAddCallback( optionMenu_2_b48, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b48,
		(XtPointer) "t_rsu" );



	/* Creation of optionMenu_2_b49 */
	optionMenu_2_b49 = XtVaCreateManagedWidget( "optionMenu_2_b49",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "shf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 117,
			NULL );
	XtAddCallback( optionMenu_2_b49, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b49,
		(XtPointer) "t_shf" );



	/* Creation of optionMenu_2_b50 */
	optionMenu_2_b50 = XtVaCreateManagedWidget( "optionMenu_2_b50",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, " Min" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 138,
			NULL );
	XtAddCallback( optionMenu_2_b50, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b50,
		(XtPointer) "t_min" );



	/* Creation of optionMenu_2_b51 */
	optionMenu_2_b51 = XtVaCreateManagedWidget( "optionMenu_2_b51",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, " Max" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 159,
			NULL );
	XtAddCallback( optionMenu_2_b51, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b51,
		(XtPointer) "t_max" );



	/* Creation of optionMenu_2_b52 */
	optionMenu_2_b52 = XtVaCreateManagedWidget( "optionMenu_2_b52",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "ddc" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 180,
			NULL );
	XtAddCallback( optionMenu_2_b52, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b52,
		(XtPointer) "t_ddc" );



	/* Creation of optionMenu_2_b53 */
	optionMenu_2_b53 = XtVaCreateManagedWidget( "optionMenu_2_b53",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "ddf" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 201,
			NULL );
	XtAddCallback( optionMenu_2_b53, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b53,
		(XtPointer) "t_ddf" );



	/* Creation of optionMenu_2_b54 */
	optionMenu_2_b54 = XtVaCreateManagedWidget( "optionMenu_2_b54",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "ddb" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 222,
			NULL );
	XtAddCallback( optionMenu_2_b54, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b54,
		(XtPointer) "t_ddb" );



	/* Creation of optionMenu_2_b55 */
	optionMenu_2_b55 = XtVaCreateManagedWidget( "optionMenu_2_b55",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "din" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 243,
			NULL );
	XtAddCallback( optionMenu_2_b55, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b55,
		(XtPointer) "t_din" );



	/* Creation of optionMenu_2_b56 */
	optionMenu_2_b56 = XtVaCreateManagedWidget( "optionMenu_2_b56",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "iin" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 264,
			NULL );
	XtAddCallback( optionMenu_2_b56, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b56,
		(XtPointer) "t_iin" );



	/* Creation of optionMenu_2_b57 */
	optionMenu_2_b57 = XtVaCreateManagedWidget( "optionMenu_2_b57",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "sbx" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 12,
			NULL );
	XtAddCallback( optionMenu_2_b57, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b57,
		(XtPointer) "t_sbx" );



	/* Creation of optionMenu_2_b58 */
	optionMenu_2_b58 = XtVaCreateManagedWidget( "optionMenu_2_b58",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "sbn" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 33,
			NULL );
	XtAddCallback( optionMenu_2_b58, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b58,
		(XtPointer) "t_sbn" );



	/* Creation of optionMenu_2_b59 */
	optionMenu_2_b59 = XtVaCreateManagedWidget( "optionMenu_2_b59",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "sbh" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 54,
			NULL );
	XtAddCallback( optionMenu_2_b59, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b59,
		(XtPointer) "t_shn" );



	/* Creation of optionMenu_2_b60 */
	optionMenu_2_b60 = XtVaCreateManagedWidget( "optionMenu_2_b60",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "sbw" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 75,
			NULL );
	XtAddCallback( optionMenu_2_b60, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b60,
		(XtPointer) "t_swl" );



	/* Creation of optionMenu_2_b61 */
	optionMenu_2_b61 = XtVaCreateManagedWidget( "optionMenu_2_b61",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "spz" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 96,
			NULL );
	XtAddCallback( optionMenu_2_b61, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b61,
		(XtPointer) "t_spz" );



	/* Creation of optionMenu_2_b62 */
	optionMenu_2_b62 = XtVaCreateManagedWidget( "optionMenu_2_b62",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "fav" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 117,
			NULL );
	XtAddCallback( optionMenu_2_b62, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b62,
		(XtPointer) "t_fav" );



	/* Creation of optionMenu_2_b63 */
	optionMenu_2_b63 = XtVaCreateManagedWidget( "optionMenu_2_b63",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "fln" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 138,
			NULL );
	XtAddCallback( optionMenu_2_b63, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b63,
		(XtPointer) "t_fln" );



	/* Creation of optionMenu_4_b1 */
	optionMenu_4_b1 = XtVaCreateManagedWidget( "optionMenu_4_b1",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, "fnr" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 159,
			NULL );
	XtAddCallback( optionMenu_4_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_4_b1,
		(XtPointer) "t_fnr" );



	/* Creation of optionMenu_2_b64 */
	optionMenu_2_b64 = XtVaCreateManagedWidget( "optionMenu_2_b64",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, " # Bad  " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 180,
			NULL );
	XtAddCallback( optionMenu_2_b64, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b64,
		(XtPointer) "t_nbd" );



	/* Creation of optionMenu_2_b65 */
	optionMenu_2_b65 = XtVaCreateManagedWidget( "optionMenu_2_b65",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, " # Good " ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 201,
			NULL );
	XtAddCallback( optionMenu_2_b65, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b65,
		(XtPointer) "t_ngd" );



	/* Creation of optionMenu_2_b66 */
	optionMenu_2_b66 = XtVaCreateManagedWidget( "optionMenu_2_b66",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, " Locate" ),
			XmNlabelType, XmSTRING,
			XmNheight, 15,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNwidth, 45,
			XmNx, 497,
			XmNy, 222,
			NULL );
	XtAddCallback( optionMenu_2_b66, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_2_b66,
		(XtPointer) "t_loc" );



	/* Creation of optionMenu_4_b2 */
	optionMenu_4_b2 = XtVaCreateManagedWidget( "optionMenu_4_b2",
			xmPushButtonWidgetClass,
			optionMenu_3,
			RES_CONVERT( XmNlabelString, " WEQ" ),
			XmNlabelType, XmSTRING,
			XmNx, 497,
			XmNy, 243,
			NULL );
	XtAddCallback( optionMenu_4_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_4_b2,
		(XtPointer) "t_weq" );



	/* Creation of optionMenu_T_TRANS */
	optionMenu_T_TRANS = XtVaCreateManagedWidget( "optionMenu_T_TRANS",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_T,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_3,
			RES_CONVERT( XmNlabelString, " " ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNmappedWhenManaged, TRUE,
			XmNy, 190,
			XmNheight, 35,
			XmNx, 0,
			XmNwidth, 104,
			NULL );


	/* Creation of rowColumn_SubTrans3 */
	rowColumn_SubTrans3 = XtVaCreateManagedWidget( "rowColumn_SubTrans3",
			xmRowColumnWidgetClass,
			rowColumn_TRANS_T,
			XmNwidth, 50,
			XmNheight, 30,
			XmNx, 80,
			XmNy, 60,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of textField_T_ARG */
	textField_T_ARG = XtVaCreateManagedWidget( "textField_T_ARG",
			xmTextFieldWidgetClass,
			rowColumn_SubTrans3,
			XmNsensitive, TRUE,
			XmNmappedWhenManaged, TRUE,
			XmNcolumns, 5,
			NULL );
	XtAddCallback( textField_T_ARG, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField_T_ARG,
		(XtPointer) NULL );
	XtAddCallback( textField_T_ARG, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_T_ARG,
		"t" );
	XtAddCallback( textField_T_ARG, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_T_ARG,
		"t" );



	/* Creation of form_XYZT */
	form_XYZT = XtVaCreateManagedWidget( "form_XYZT",
			xmFormWidgetClass,
			form_Context,
			XmNwidth, 200,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of rowColumn_XYZT_Regridding */
	rowColumn_XYZT_Regridding = XtVaCreateManagedWidget( "rowColumn_XYZT_Regridding",
			xmRowColumnWidgetClass,
			form_XYZT,
			XmNwidth, 50,
			XmNheight, 20,
			XmNx, 75,
			XmNy, 109,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 0,
			NULL );


	/* Creation of toggleButton_Regridding */
	toggleButton_Regridding = XtVaCreateManagedWidget( "toggleButton_Regridding",
			xmToggleButtonWidgetClass,
			rowColumn_XYZT_Regridding,
			XmNx, 10,
			XmNy, 40,
			XmNwidth, 20,
			XmNheight, 20,
			RES_CONVERT( XmNlabelString, "" ),
			NULL );
	XtAddCallback( toggleButton_Regridding, XmNvalueChangedCallback,
		(XtCallbackProc) JC_FixedToggle_CB,
		(XtPointer) NULL );



	/* Creation of rowColumn_subRegridding */
	rowColumn_subRegridding = XtVaCreateManagedWidget( "rowColumn_subRegridding",
			xmRowColumnWidgetClass,
			rowColumn_XYZT_Regridding,
			NULL );


	/* Creation of pushButton_SelectRegridding */
	pushButton_SelectRegridding = XtVaCreateManagedWidget( "pushButton_SelectRegridding",
			xmPushButtonWidgetClass,
			rowColumn_subRegridding,
			RES_CONVERT( XmNlabelString, "Set Regridding..." ),
			NULL );
	XtAddCallback( pushButton_SelectRegridding, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton_SelectRegridding,
		(XtPointer) NULL );



	/* Creation of rowColumn_XYZT_X */
	rowColumn_XYZT_X = XtVaCreateManagedWidget( "rowColumn_XYZT_X",
			xmRowColumnWidgetClass,
			form_XYZT,
			XmNwidth, 50,
			XmNheight, 20,
			XmNx, 290,
			XmNy, 10,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNorientation, XmHORIZONTAL,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 63,
			NULL );


	/* Creation of toggleButton_X */
	toggleButton_X = XtVaCreateManagedWidget( "toggleButton_X",
			xmToggleButtonWidgetClass,
			rowColumn_XYZT_X,
			RES_CONVERT( XmNlabelString, "" ),
			NULL );
	XtAddCallback( toggleButton_X, XmNvalueChangedCallback,
		(XtCallbackProc) JC_FixedToggle_CB,
		(XtPointer) NULL );



	/* Creation of optionMenu_Xp */
	optionMenu_Xp_shell = XtVaCreatePopupShell ("optionMenu_Xp_shell",
			xmMenuShellWidgetClass, rowColumn_XYZT_X,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_Xp = XtVaCreateWidget( "optionMenu_Xp",
			xmRowColumnWidgetClass,
			optionMenu_Xp_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_Xp_Longitude */
	optionMenu_Xp_Longitude = XtVaCreateManagedWidget( "optionMenu_Xp_Longitude",
			xmPushButtonWidgetClass,
			optionMenu_Xp,
			RES_CONVERT( XmNlabelString, "___________" ),
			NULL );
	XtAddCallback( optionMenu_Xp_Longitude, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Xp_Longitude,
		(XtPointer) NULL );



	/* Creation of optionMenu_Xp_Index */
	optionMenu_Xp_Index = XtVaCreateManagedWidget( "optionMenu_Xp_Index",
			xmPushButtonWidgetClass,
			optionMenu_Xp,
			RES_CONVERT( XmNlabelString, "X Index" ),
			NULL );
	XtAddCallback( optionMenu_Xp_Index, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Xp_Index,
		(XtPointer) NULL );



	/* Creation of optionMenu_X */
	optionMenu_X = XtVaCreateManagedWidget( "optionMenu_X",
			xmRowColumnWidgetClass,
			rowColumn_XYZT_X,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_Xp,
			NULL );


	/* Creation of rowColumn_XYZT_Y */
	rowColumn_XYZT_Y = XtVaCreateManagedWidget( "rowColumn_XYZT_Y",
			xmRowColumnWidgetClass,
			form_XYZT,
			XmNwidth, 50,
			XmNheight, 20,
			XmNx, 75,
			XmNy, 109,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 117,
			NULL );


	/* Creation of toggleButton_Y */
	toggleButton_Y = XtVaCreateManagedWidget( "toggleButton_Y",
			xmToggleButtonWidgetClass,
			rowColumn_XYZT_Y,
			RES_CONVERT( XmNlabelString, "" ),
			NULL );
	XtAddCallback( toggleButton_Y, XmNvalueChangedCallback,
		(XtCallbackProc) JC_FixedToggle_CB,
		(XtPointer) NULL );



	/* Creation of optionMenu_Yp */
	optionMenu_Yp_shell = XtVaCreatePopupShell ("optionMenu_Yp_shell",
			xmMenuShellWidgetClass, rowColumn_XYZT_Y,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_Yp = XtVaCreateWidget( "optionMenu_Yp",
			xmRowColumnWidgetClass,
			optionMenu_Yp_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_Yp_Latitude */
	optionMenu_Yp_Latitude = XtVaCreateManagedWidget( "optionMenu_Yp_Latitude",
			xmPushButtonWidgetClass,
			optionMenu_Yp,
			RES_CONVERT( XmNlabelString, "___________" ),
			NULL );
	XtAddCallback( optionMenu_Yp_Latitude, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Yp_Latitude,
		(XtPointer) NULL );



	/* Creation of optionMenu_Yp_Index */
	optionMenu_Yp_Index = XtVaCreateManagedWidget( "optionMenu_Yp_Index",
			xmPushButtonWidgetClass,
			optionMenu_Yp,
			RES_CONVERT( XmNlabelString, "Y Index" ),
			NULL );
	XtAddCallback( optionMenu_Yp_Index, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Yp_Index,
		(XtPointer) NULL );



	/* Creation of optionMenu_Y */
	optionMenu_Y = XtVaCreateManagedWidget( "optionMenu_Y",
			xmRowColumnWidgetClass,
			rowColumn_XYZT_Y,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_Yp,
			NULL );


	/* Creation of rowColumn_XYZT_Z */
	rowColumn_XYZT_Z = XtVaCreateManagedWidget( "rowColumn_XYZT_Z",
			xmRowColumnWidgetClass,
			form_XYZT,
			XmNwidth, 50,
			XmNheight, 20,
			XmNx, 75,
			XmNy, 109,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 171,
			NULL );


	/* Creation of toggleButton_Z */
	toggleButton_Z = XtVaCreateManagedWidget( "toggleButton_Z",
			xmToggleButtonWidgetClass,
			rowColumn_XYZT_Z,
			RES_CONVERT( XmNlabelString, "" ),
			NULL );
	XtAddCallback( toggleButton_Z, XmNvalueChangedCallback,
		(XtCallbackProc) JC_FixedToggle_CB,
		(XtPointer) NULL );



	/* Creation of optionMenu_Zp */
	optionMenu_Zp_shell = XtVaCreatePopupShell ("optionMenu_Zp_shell",
			xmMenuShellWidgetClass, rowColumn_XYZT_Z,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_Zp = XtVaCreateWidget( "optionMenu_Zp",
			xmRowColumnWidgetClass,
			optionMenu_Zp_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_Zp_Depth */
	optionMenu_Zp_Depth = XtVaCreateManagedWidget( "optionMenu_Zp_Depth",
			xmPushButtonWidgetClass,
			optionMenu_Zp,
			RES_CONVERT( XmNlabelString, "___________" ),
			NULL );
	XtAddCallback( optionMenu_Zp_Depth, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Zp_Depth,
		(XtPointer) NULL );



	/* Creation of optionMenu_Zp_Index */
	optionMenu_Zp_Index = XtVaCreateManagedWidget( "optionMenu_Zp_Index",
			xmPushButtonWidgetClass,
			optionMenu_Zp,
			RES_CONVERT( XmNlabelString, "Z Index" ),
			NULL );
	XtAddCallback( optionMenu_Zp_Index, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Zp_Index,
		(XtPointer) NULL );



	/* Creation of optionMenu_Z */
	optionMenu_Z = XtVaCreateManagedWidget( "optionMenu_Z",
			xmRowColumnWidgetClass,
			rowColumn_XYZT_Z,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_Zp,
			NULL );


	/* Creation of rowColumn_XYZT_T */
	rowColumn_XYZT_T = XtVaCreateManagedWidget( "rowColumn_XYZT_T",
			xmRowColumnWidgetClass,
			form_XYZT,
			XmNwidth, 50,
			XmNheight, 20,
			XmNx, 75,
			XmNy, 109,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 225,
			NULL );


	/* Creation of toggleButton_T */
	toggleButton_T = XtVaCreateManagedWidget( "toggleButton_T",
			xmToggleButtonWidgetClass,
			rowColumn_XYZT_T,
			RES_CONVERT( XmNlabelString, "" ),
			NULL );
	XtAddCallback( toggleButton_T, XmNvalueChangedCallback,
		(XtCallbackProc) JC_FixedToggle_CB,
		(XtPointer) NULL );



	/* Creation of optionMenu_Tp */
	optionMenu_Tp_shell = XtVaCreatePopupShell ("optionMenu_Tp_shell",
			xmMenuShellWidgetClass, rowColumn_XYZT_T,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_Tp = XtVaCreateWidget( "optionMenu_Tp",
			xmRowColumnWidgetClass,
			optionMenu_Tp_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_Tp_Calendar */
	optionMenu_Tp_Calendar = XtVaCreateManagedWidget( "optionMenu_Tp_Calendar",
			xmPushButtonWidgetClass,
			optionMenu_Tp,
			RES_CONVERT( XmNlabelString, "___________" ),
			NULL );
	XtAddCallback( optionMenu_Tp_Calendar, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Tp_Calendar,
		(XtPointer) NULL );



	/* Creation of optionMenu_Tp_Climatology */
	optionMenu_Tp_Climatology = XtVaCreateManagedWidget( "optionMenu_Tp_Climatology",
			xmPushButtonWidgetClass,
			optionMenu_Tp,
			RES_CONVERT( XmNlabelString, "Climatology" ),
			NULL );
	XtAddCallback( optionMenu_Tp_Climatology, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Tp_Climatology,
		(XtPointer) NULL );



	/* Creation of optionMenu_Tp_Model */
	optionMenu_Tp_Model = XtVaCreateManagedWidget( "optionMenu_Tp_Model",
			xmPushButtonWidgetClass,
			optionMenu_Tp,
			RES_CONVERT( XmNlabelString, "Time (raw)" ),
			NULL );
	XtAddCallback( optionMenu_Tp_Model, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Tp_Model,
		(XtPointer) NULL );



	/* Creation of optionMenu_Tp_Index */
	optionMenu_Tp_Index = XtVaCreateManagedWidget( "optionMenu_Tp_Index",
			xmPushButtonWidgetClass,
			optionMenu_Tp,
			RES_CONVERT( XmNlabelString, "T Index" ),
			NULL );
	XtAddCallback( optionMenu_Tp_Index, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_Tp_Index,
		(XtPointer) NULL );



	/* Creation of optionMenu_T */
	optionMenu_T = XtVaCreateManagedWidget( "optionMenu_T",
			xmRowColumnWidgetClass,
			rowColumn_XYZT_T,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_Tp,
			NULL );


	/* Creation of label_RegriddingStatus */
	label_RegriddingStatus = XtVaCreateManagedWidget( "label_RegriddingStatus",
			xmLabelWidgetClass,
			form_XYZT,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 5,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightOffset, 5,
			XmNrightPosition, 100,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_XYZT_Regridding,
			NULL );


	/* Creation of form_Region */
	form_Region = XtVaCreateManagedWidget( "form_Region",
			xmFormWidgetClass,
			form_Context,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNrightAttachment, XmATTACH_WIDGET,
			XmNrightWidget, form_Transforms,
			XmNleftWidget, form_XYZT,
			XmNfractionBase, 8,
			XmNtopOffset, 40,
			NULL );


	/* Creation of textField_X_LO */
	textField_X_LO = XtVaCreateManagedWidget( "textField_X_LO",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNtopOffset, 30,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			NULL );
	XtAddCallback( textField_X_LO, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"x_l" );
	XtAddCallback( textField_X_LO, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"x_l" );



	/* Creation of textField_Y_LO */
	textField_Y_LO = XtVaCreateManagedWidget( "textField_Y_LO",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNtopOffset, 84,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			NULL );
	XtAddCallback( textField_Y_LO, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"y_l" );
	XtAddCallback( textField_Y_LO, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"y_l" );



	/* Creation of textField_Z_LO */
	textField_Z_LO = XtVaCreateManagedWidget( "textField_Z_LO",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 138,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			NULL );
	XtAddCallback( textField_Z_LO, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"z_l" );
	XtAddCallback( textField_Z_LO, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"z_l" );



	/* Creation of scrollBar_X_LO */
	scrollBar_X_LO = XtVaCreateManagedWidget( "scrollBar_X_LO",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_X_LO,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_X_LO, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_X_LO,
		"x_l" );
	XtAddCallback( scrollBar_X_LO, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_X_LO,
		"x_l" );



	/* Creation of scrollBar_Y_LO */
	scrollBar_Y_LO = XtVaCreateManagedWidget( "scrollBar_Y_LO",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopOffset, 0,
			XmNtopWidget, textField_Y_LO,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_Y_LO, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_Y_LO,
		"y_l" );
	XtAddCallback( scrollBar_Y_LO, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_Y_LO,
		"y_l" );



	/* Creation of scrollBar_Z_LO */
	scrollBar_Z_LO = XtVaCreateManagedWidget( "scrollBar_Z_LO",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 0,
			XmNtopWidget, textField_Z_LO,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_Z_LO, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_Z_LO,
		"z_l" );
	XtAddCallback( scrollBar_Z_LO, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_Z_LO,
		"z_l" );



	/* Creation of textField_T_LO */
	textField_T_LO = XtVaCreateManagedWidget( "textField_T_LO",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 192,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			NULL );
	XtAddCallback( textField_T_LO, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"t_l" );
	XtAddCallback( textField_T_LO, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"t_l" );



	/* Creation of scrollBar_T_LO */
	scrollBar_T_LO = XtVaCreateManagedWidget( "scrollBar_T_LO",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopOffset, 0,
			XmNtopWidget, textField_T_LO,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 4,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_T_LO, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_T_LO,
		"t_l" );
	XtAddCallback( scrollBar_T_LO, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_T_LO,
		"t_l" );



	/* Creation of textField_X_PT */
	textField_X_PT = XtVaCreateManagedWidget( "textField_X_PT",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNtopOffset, 30,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 2,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			NULL );
	XtAddCallback( textField_X_PT, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"x_p" );
	XtAddCallback( textField_X_PT, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"x_p" );



	/* Creation of textField_Y_PT */
	textField_Y_PT = XtVaCreateManagedWidget( "textField_Y_PT",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 84,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 2,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			NULL );
	XtAddCallback( textField_Y_PT, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"y_p" );
	XtAddCallback( textField_Y_PT, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"y_p" );



	/* Creation of textField_Z_PT */
	textField_Z_PT = XtVaCreateManagedWidget( "textField_Z_PT",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 138,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 2,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			NULL );
	XtAddCallback( textField_Z_PT, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"z_p" );
	XtAddCallback( textField_Z_PT, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"z_p" );



	/* Creation of scrollBar_X_PT */
	scrollBar_X_PT = XtVaCreateManagedWidget( "scrollBar_X_PT",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_X_PT,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 2,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_X_PT, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_X_PT,
		"x_p" );
	XtAddCallback( scrollBar_X_PT, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_X_PT,
		"x_p" );



	/* Creation of scrollBar_Y_PT */
	scrollBar_Y_PT = XtVaCreateManagedWidget( "scrollBar_Y_PT",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_Y_PT,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 2,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_Y_PT, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_Y_PT,
		"y_p" );
	XtAddCallback( scrollBar_Y_PT, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_Y_PT,
		"y_p" );



	/* Creation of scrollBar_Z_PT */
	scrollBar_Z_PT = XtVaCreateManagedWidget( "scrollBar_Z_PT",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_Z_PT,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 2,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_Z_PT, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_Z_PT,
		"z_p" );
	XtAddCallback( scrollBar_Z_PT, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_Z_PT,
		"z_p" );



	/* Creation of textField_T_PT */
	textField_T_PT = XtVaCreateManagedWidget( "textField_T_PT",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 192,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 2,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			NULL );
	XtAddCallback( textField_T_PT, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"t_p" );
	XtAddCallback( textField_T_PT, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"t_p" );



	/* Creation of scrollBar_T_PT */
	scrollBar_T_PT = XtVaCreateManagedWidget( "scrollBar_T_PT",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_T_PT,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 2,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 6,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_T_PT, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_T_PT,
		"t_p" );
	XtAddCallback( scrollBar_T_PT, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_T_PT,
		"t_p" );



	/* Creation of textField_X_HI */
	textField_X_HI = XtVaCreateManagedWidget( "textField_X_HI",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNtopOffset, 30,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 4,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			NULL );
	XtAddCallback( textField_X_HI, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"x_h" );
	XtAddCallback( textField_X_HI, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"x_h" );



	/* Creation of textField_Y_HI */
	textField_Y_HI = XtVaCreateManagedWidget( "textField_Y_HI",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNtopOffset, 84,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 4,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			NULL );
	XtAddCallback( textField_Y_HI, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"y_h" );
	XtAddCallback( textField_Y_HI, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"y_h" );



	/* Creation of textField_Z_HI */
	textField_Z_HI = XtVaCreateManagedWidget( "textField_Z_HI",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 138,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 4,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			NULL );
	XtAddCallback( textField_Z_HI, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"z_h" );
	XtAddCallback( textField_Z_HI, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"z_h" );



	/* Creation of scrollBar_Z_HI */
	scrollBar_Z_HI = XtVaCreateManagedWidget( "scrollBar_Z_HI",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_Z_HI,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 4,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_Z_HI, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_Z_HI,
		"z_h" );
	XtAddCallback( scrollBar_Z_HI, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_Z_HI,
		"z_h" );



	/* Creation of scrollBar_Y_HI */
	scrollBar_Y_HI = XtVaCreateManagedWidget( "scrollBar_Y_HI",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_Y_HI,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 4,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_Y_HI, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_Y_HI,
		"y_h" );
	XtAddCallback( scrollBar_Y_HI, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_Y_HI,
		"y_h" );



	/* Creation of scrollBar_X_HI */
	scrollBar_X_HI = XtVaCreateManagedWidget( "scrollBar_X_HI",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_X_HI,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 4,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_X_HI, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_X_HI,
		"x_h" );
	XtAddCallback( scrollBar_X_HI, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_X_HI,
		"x_h" );



	/* Creation of textField_T_HI */
	textField_T_HI = XtVaCreateManagedWidget( "textField_T_HI",
			xmTextFieldWidgetClass,
			form_Region,
			XmNwidth, 80,
			XmNvalue, "",
			XmNbottomAttachment, XmATTACH_NONE,
			XmNbottomOffset, 0,
			XmNbottomWidget, NULL,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 192,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftPosition, 4,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			NULL );
	XtAddCallback( textField_T_HI, XmNactivateCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"t_h" );
	XtAddCallback( textField_T_HI, XmNlosingFocusCallback,
		(XtCallbackProc) JC_XYZTTextField_Verify_CB,
		"t_h" );



	/* Creation of scrollBar_T_HI */
	scrollBar_T_HI = XtVaCreateManagedWidget( "scrollBar_T_HI",
			xmScrollBarWidgetClass,
			form_Region,
			XmNprocessingDirection, XmMAX_ON_RIGHT,
			XmNtopWidget, textField_T_HI,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 4,
			XmNorientation, XmHORIZONTAL,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 8,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );
	XtAddCallback( scrollBar_T_HI, XmNdragCallback,
		(XtCallbackProc) dragCB_scrollBar_T_HI,
		"t_h" );
	XtAddCallback( scrollBar_T_HI, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrollBar_T_HI,
		"t_h" );



	/* Creation of optionMenu_p4 */
	optionMenu_p4_shell = XtVaCreatePopupShell ("optionMenu_p4_shell",
			xmMenuShellWidgetClass, form_Context,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p4 = XtVaCreateWidget( "optionMenu_p4",
			xmRowColumnWidgetClass,
			optionMenu_p4_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 0,
			XmNy, -156,
			XmNheight, 26,
			XmNwidth, 155,
			NULL );


	/* Creation of optionMenu_p_b4 */
	optionMenu_p_b4 = XtVaCreateManagedWidget( "optionMenu_p_b4",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "Point" ),
			XmNx, 2,
			XmNy, 4,
			XmNheight, 26,
			XmNmappedWhenManaged, TRUE,
			XmNwidth, 155,
			NULL );
	XtAddCallback( optionMenu_p_b4, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p_b4,
		"0" );



	/* Creation of optionMenu_p4_b2 */
	optionMenu_p4_b2 = XtVaCreateManagedWidget( "optionMenu_p4_b2",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "X Line" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b2,
		"1" );



	/* Creation of optionMenu_p4_b3 */
	optionMenu_p4_b3 = XtVaCreateManagedWidget( "optionMenu_p4_b3",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "Y Line" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b3, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b3,
		"2" );



	/* Creation of optionMenu_p4_b4 */
	optionMenu_p4_b4 = XtVaCreateManagedWidget( "optionMenu_p4_b4",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "Z Line" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b4, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b4,
		"3" );



	/* Creation of optionMenu_p4_b5 */
	optionMenu_p4_b5 = XtVaCreateManagedWidget( "optionMenu_p4_b5",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "T Line" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b5, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b5,
		"4" );



	/* Creation of optionMenu_p4_b6 */
	optionMenu_p4_b6 = XtVaCreateManagedWidget( "optionMenu_p4_b6",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "XY Plane" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b6, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b6,
		"5" );



	/* Creation of optionMenu_p4_b7 */
	optionMenu_p4_b7 = XtVaCreateManagedWidget( "optionMenu_p4_b7",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "XZ Plane" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b7, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b7,
		"6" );



	/* Creation of optionMenu_p4_b8 */
	optionMenu_p4_b8 = XtVaCreateManagedWidget( "optionMenu_p4_b8",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "XT Plane" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b8, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b8,
		"7" );



	/* Creation of optionMenu_p4_b9 */
	optionMenu_p4_b9 = XtVaCreateManagedWidget( "optionMenu_p4_b9",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "YZ Plane" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b9, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b9,
		"8" );



	/* Creation of optionMenu_p4_b10 */
	optionMenu_p4_b10 = XtVaCreateManagedWidget( "optionMenu_p4_b10",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "YT Plane" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b10, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b10,
		"9" );



	/* Creation of optionMenu_p4_b11 */
	optionMenu_p4_b11 = XtVaCreateManagedWidget( "optionMenu_p4_b11",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "ZT Plane" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b11, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b11,
		"10" );



	/* Creation of optionMenu_p4_b12 */
	optionMenu_p4_b12 = XtVaCreateManagedWidget( "optionMenu_p4_b12",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "XYZ Volume" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b12, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b12,
		"11" );



	/* Creation of optionMenu_p4_b13 */
	optionMenu_p4_b13 = XtVaCreateManagedWidget( "optionMenu_p4_b13",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "XYT Volume" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b13, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b13,
		"12" );



	/* Creation of optionMenu_p4_b14 */
	optionMenu_p4_b14 = XtVaCreateManagedWidget( "optionMenu_p4_b14",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "XZT Volume" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b14, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b14,
		"13" );



	/* Creation of optionMenu_p4_b15 */
	optionMenu_p4_b15 = XtVaCreateManagedWidget( "optionMenu_p4_b15",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "YZT Volume" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b15, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b15,
		"14" );



	/* Creation of optionMenu_p4_b16 */
	optionMenu_p4_b16 = XtVaCreateManagedWidget( "optionMenu_p4_b16",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "XYZT (4D)" ),
			XmNx, 2,
			XmNy, 4,
			NULL );
	XtAddCallback( optionMenu_p4_b16, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p4_b16,
		"15" );



	/* Creation of optionMenu_Geometry */
	optionMenu_Geometry = XtVaCreateManagedWidget( "optionMenu_Geometry",
			xmRowColumnWidgetClass,
			form_Context,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p4,
			RES_CONVERT( XmNlabelString, " " ),
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, -50,
			XmNleftPosition, 50,
			NULL );


	/* Creation of frame_plot */
	frame_plot = XtVaCreateManagedWidget( "frame_plot",
			xmFrameWidgetClass,
			form1,
			XmNrightOffset, 5,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, frame_context,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of label10 */
	label10 = XtVaCreateManagedWidget( "label10",
			xmLabelWidgetClass,
			frame_plot,
			RES_CONVERT( XmNlabelString, "Plotting" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_plot */
	form_plot = XtVaCreateManagedWidget( "form_plot",
			xmFormWidgetClass,
			frame_plot,
			NULL );


	/* Creation of rowColumn_PlotRadios */
	rowColumn_PlotRadios = XtVaCreateManagedWidget( "rowColumn_PlotRadios",
			xmRowColumnWidgetClass,
			form_plot,
			XmNorientation, XmHORIZONTAL,
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 0,
			XmNtopAttachment, XmATTACH_FORM,
			XmNpacking, XmPACK_TIGHT,
			XmNradioBehavior, TRUE,
			NULL );


	/* Creation of toggleButton_Line */
	toggleButton_Line = XtVaCreateManagedWidget( "toggleButton_Line",
			xmToggleButtonWidgetClass,
			rowColumn_PlotRadios,
			RES_CONVERT( XmNlabelString, "Line" ),
			NULL );
	XtAddCallback( toggleButton_Line, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_Line,
		(XtPointer) NULL );



	/* Creation of toggleButton_Scatter */
	toggleButton_Scatter = XtVaCreateManagedWidget( "toggleButton_Scatter",
			xmToggleButtonWidgetClass,
			rowColumn_PlotRadios,
			RES_CONVERT( XmNlabelString, "Scatter" ),
			NULL );
	XtAddCallback( toggleButton_Scatter, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_Scatter,
		(XtPointer) NULL );



	/* Creation of toggleButton_Shade */
	toggleButton_Shade = XtVaCreateManagedWidget( "toggleButton_Shade",
			xmToggleButtonWidgetClass,
			rowColumn_PlotRadios,
			RES_CONVERT( XmNlabelString, "Shade" ),
			NULL );
	XtAddCallback( toggleButton_Shade, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_Shade,
		(XtPointer) NULL );



	/* Creation of toggleButton_Contour */
	toggleButton_Contour = XtVaCreateManagedWidget( "toggleButton_Contour",
			xmToggleButtonWidgetClass,
			rowColumn_PlotRadios,
			RES_CONVERT( XmNlabelString, "Contour" ),
			NULL );
	XtAddCallback( toggleButton_Contour, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_Contour,
		(XtPointer) NULL );



	/* Creation of toggleButton_Fill */
	toggleButton_Fill = XtVaCreateManagedWidget( "toggleButton_Fill",
			xmToggleButtonWidgetClass,
			rowColumn_PlotRadios,
			RES_CONVERT( XmNlabelString, "Fill" ),
			NULL );
	XtAddCallback( toggleButton_Fill, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_Fill,
		(XtPointer) NULL );



	/* Creation of toggleButton_Vector */
	toggleButton_Vector = XtVaCreateManagedWidget( "toggleButton_Vector",
			xmToggleButtonWidgetClass,
			rowColumn_PlotRadios,
			RES_CONVERT( XmNlabelString, "Vector" ),
			NULL );
	XtAddCallback( toggleButton_Vector, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_Vector,
		(XtPointer) NULL );



	/* Creation of rowColumn_PlotButtons */
	rowColumn_PlotButtons = XtVaCreateManagedWidget( "rowColumn_PlotButtons",
			xmRowColumnWidgetClass,
			form_plot,
			XmNorientation, XmHORIZONTAL,
			XmNpacking, XmPACK_COLUMN,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNleftWidget, rowColumn_PlotRadios,
			XmNtopOffset, 0,
			XmNentryAlignment, XmALIGNMENT_CENTER,
			NULL );


	/* Creation of pushButton_Plot */
	pushButton_Plot = XtVaCreateManagedWidget( "pushButton_Plot",
			xmPushButtonWidgetClass,
			rowColumn_PlotButtons,
			RES_CONVERT( XmNlabelString, "Plot" ),
			NULL );
	XtAddCallback( pushButton_Plot, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton_Plot,
		(XtPointer) NULL );



	/* Creation of pushButton_Overlay */
	pushButton_Overlay = XtVaCreateManagedWidget( "pushButton_Overlay",
			xmPushButtonWidgetClass,
			rowColumn_PlotButtons,
			RES_CONVERT( XmNlabelString, "Overlay" ),
			NULL );
	XtAddCallback( pushButton_Overlay, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton_Overlay,
		(XtPointer) NULL );



	/* Creation of frame_map */
	frame_map = XtVaCreateManagedWidget( "frame_map",
			xmFrameWidgetClass,
			form1,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNrightOffset, 5,
			XmNmappedWhenManaged, FALSE,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNwidth, 715,
			XmNtopOffset, 5,
			XmNtopWidget, frame_plot,
			XmNbottomOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of bulletinBoard9 */
	bulletinBoard9 = XtVaCreateManagedWidget( "bulletinBoard9",
			xmBulletinBoardWidgetClass,
			frame_map,
			NULL );


	/* Creation of drawingArea1 */
	drawingArea1 = XtVaCreateManagedWidget( "drawingArea1",
			xmDrawingAreaWidgetClass,
			bulletinBoard9,
			RES_CONVERT( XmNtranslations, Map_translations ),
			XmNheight, 216,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNwidth, 627,
			NULL );
	XtAddCallback( drawingArea1, XmNexposeCallback,
		(XtCallbackProc) exposeCB_drawingArea1,
		(XtPointer) NULL );



	/* Creation of StartupMessage */
	StartupMessage = XtVaCreateManagedWidget( "StartupMessage",
			xmLabelWidgetClass,
			form1,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, frame_Data,
			RES_CONVERT( XmNlabelString, "\
Use the \"File\" menu to open a dataset." ),
			XmNleftAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNtopOffset, 100,
			NULL );

	XtVaSetValues(Ferret_app_menu1,
			XmNmenuHelpWidget, helpCascade,
			NULL );

	XtVaSetValues(scriptsCascade,
			XmNpositionIndex, 4,
			NULL );

	XtVaSetValues(viewCascade,
			XmNpositionIndex, 2,
			NULL );

	XtVaSetValues(optionsCascade,
			XmNpositionIndex, 3,
			NULL );

	XtVaSetValues(pushButton_Clone,
			XmNpositionIndex, 3,
			NULL );

	XtVaSetValues(textField_Variable,
			XmNpositionIndex, 0,
			NULL );

	XtVaSetValues(textField_Dataset,
			XmNpositionIndex, 2,
			NULL );

	XtVaSetValues(label1,
			XmNpositionIndex, 1,
			NULL );

	XtVaSetValues(drawingArea1,
			XmNbackgroundPixmap, UxConvertPixmap( "map_pn_final.xpm" ),
			NULL );



	return ( FerretMainWd );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_FerretMainWd( swidget _UxUxParent )
{
	Widget                  rtrn;
	static int		_Uxinit = 0;

	UxParent = _UxUxParent;

	if ( ! _Uxinit )
	{
		static XtActionsRec	_Uxactions[] = {
			{ "Map_Btn1Motion", (XtActionProc) action_Map_Btn1Motion },
			{ "Map_Btn1Up", (XtActionProc) action_Map_Btn1Up },
			{ "Map_Btn1Down", (XtActionProc) action_Map_Btn1Down }};

		XtAppAddActions( UxAppContext,
				_Uxactions,
				XtNumber(_Uxactions) );

		XmRepTypeInstallTearOffModelConverter();
		_Uxinit = 1;
	}

	{
		Display *display;
		Screen *screen;
		Window window;
		int widthMM, widthPix;
		float widthIN, screenRez;
		Atom wm_delete_window;
		Widget shell;
		rtrn = _Uxbuild_FerretMainWd();

		/*shell = XtParent(rtrn);
			XtVaSetValues(shell,
				XmNdeleteResponse, XmDO_NOTHING, 
				NULL);*/
		
			/* add the close window code */
			wm_delete_window = XmInternAtom(
						XtDisplay(rtrn),
						"WM_DELETE_WINDOW",
						False);
			XmAddWMProtocols(rtrn, &wm_delete_window, 1);
			XmAddWMProtocolCallback(rtrn, wm_delete_window, CloseWinCB, rtrn);
		
			InitFerretStructs();
			
			display = XtDisplay(UxGetWidget(drawingArea1));
			screen = DefaultScreenOfDisplay(display);
			window = RootWindowOfScreen(screen);
		
			/* get the resolution of the device */
			widthMM = DisplayWidthMM(display, DefaultScreen(display));
			widthPix = DisplayWidth(display, DefaultScreen(display));
			widthIN = (float)widthMM/25.4;
			screenRez = (float)widthPix/widthIN;
		
			if (screenRez >= 91)
				gHiRez = True;
			else
				gHiRez = False;
		
			/* configure and create a graphics context */ 
			gcv.foreground = WhitePixelOfScreen(screen);
			gcv.line_width = 1;
						
			gc = XCreateGC(display, window, GCForeground+GCLineWidth, &gcv);
		
			/* install a graphics context into user data of widget */
			XtVaSetValues(UxGetWidget(drawingArea1),
				XmNuserData, gc, 
				NULL);
		
			drGcv.background = BlackPixelOfScreen(screen);
			drGcv.foreground = WhitePixelOfScreen(screen);
			drGcv.fill_style = FillStippled;
			drGcv.fill_rule = WindingRule;
			drGcv.stipple = XCreatePixmapFromBitmapData(display, window,
				stippleBits, 2, 2, BlackPixelOfScreen(screen), 
				WhitePixelOfScreen(screen), 1);
			drGc = XCreateGC(display, window, GCForeground+GCBackground+
				GCFillStyle+GCFillRule+GCStipple, &drGcv);	
		
			InitGlobalWidgets();
			InitPixmaps();
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		/* set the hi rez size */
		if (gHiRez) {
			Dimension width, height;
				
			XtVaGetValues(UxGetWidget(form1),
				XmNwidth, &width,
				XmNheight, &height,
				NULL);
			width = 1.15 * width;
			height = 1.2 * height;
				
			XtVaSetValues(UxGetWidget(form1),
				XmNwidth, width,
				XmNheight, height,
				NULL);
		}
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

