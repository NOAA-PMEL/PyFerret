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
	PlotOptions.c

       Associated Header file: PlotOptions.h
       Associated Resource file: PlotOptions.rf
*******************************************************************************/

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

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "ferret_structures.h"

#include "lines_1.xpm"
#include "lines_1_ins.xpm"
#include "lines_2.xpm"
#include "lines_3.xpm"
#include "lines_4.xpm"
#include "lines_5.xpm"
#include "lines_6.xpm"
#include "lines_single.xpm"
#include "lines_double.xpm"
#include "lines_triple.xpm"
#include "lines_single_ins.xpm"
#include "lines_double_ins.xpm"
#include "lines_triple_ins.xpm"
#ifdef FULL_GUI_VERSION
#include "sym_1.xpm"
#include "sym_2.xpm"
#include "sym_3.xpm"
#include "sym_4.xpm"
#include "sym_5.xpm"
#include "sym_6.xpm"
#include "sym_7.xpm"
#include "sym_8.xpm"
#include "sym_9.xpm"
#include "sym_10.xpm"
#include "sym_11.xpm"
#include "sym_12.xpm"
#include "sym_13.xpm"
#include "sym_14.xpm"
#include "sym_15.xpm"
#include "sym_16.xpm"
#include "sym_17.xpm"
#include "sym_18.xpm"
#include "sym_19.xpm"
#include "sym_20.xpm"
#include "sym_21.xpm"
#include "sym_22.xpm"
#include "sym_23.xpm"
#include "sym_24.xpm"
#include "sym_25.xpm"
#include "sym_26.xpm"
#include "sym_27.xpm"
#include "sym_28.xpm"
#include "sym_29.xpm"
#include "sym_30.xpm"
#include "sym_31.xpm"
#include "sym_32.xpm"
#include "sym_33.xpm"
#include "sym_34.xpm"
#include "sym_35.xpm"
#include "sym_36.xpm"
#include "sym_37.xpm"
#include "sym_38.xpm"
#include "sym_39.xpm"
#include "sym_40.xpm"
#include "sym_41.xpm"
#include "sym_42.xpm"
#include "sym_43.xpm"
#include "sym_44.xpm"
#include "sym_45.xpm"
#include "sym_46.xpm"
#include "sym_47.xpm"
#include "sym_48.xpm"
#include "sym_49.xpm"
#include "sym_50.xpm"
#include "sym_51.xpm"
#include "sym_52.xpm"
#include "sym_53.xpm"
#include "sym_54.xpm"
#include "sym_55.xpm"
#include "sym_56.xpm"
#include "sym_57.xpm"
#include "sym_58.xpm"
#include "sym_59.xpm"
#include "sym_60.xpm"
#include "sym_61.xpm"
#include "sym_62.xpm"
#include "sym_63.xpm"
#include "sym_64.xpm"
#include "sym_65.xpm"
#include "sym_66.xpm"
#include "sym_67.xpm"
#include "sym_68.xpm"
#include "sym_69.xpm"
#include "sym_70.xpm"
#include "sym_71.xpm"
#include "sym_72.xpm"
#include "sym_73.xpm"
#include "sym_74.xpm"
#include "sym_75.xpm"
#include "sym_76.xpm"
#include "sym_77.xpm"
#include "sym_78.xpm"
#include "sym_79.xpm"
#include "sym_80.xpm"
#include "sym_81.xpm"
#include "sym_82.xpm"
#include "sym_83.xpm"
#include "sym_84.xpm"
#include "sym_85.xpm"
#include "sym_86.xpm"
#include "sym_87.xpm"
#include "sym_88.xpm"
#endif

swidget create_PlotOptions(swidget UxParent);
static void InitPixmaps();
void SetInitialPOState(void);
extern Pixmap GetPixmapFromData(char **inData);

/* visual state of plot options */
void PlotOptions2Interface(void);

/* functions that read state of interface and store */
int Update1DOptionsCB(void);
int Update2DOptionsCB(void);
int UpdateVectorOptionsCB(void);
void UpdateGeneralOptionsCB(void);
int UpdateLineStyleCB(Widget wid, XtPointer client_data,
	       XtPointer cbs);
int UpdateLineSymbolCB(Widget wid, XtPointer client_data,
	       XtPointer cbs);
static void InitArrays(void);
static void DisableThickBtn(void);
static void EnableThickBtn(void);

/* external functions */
extern char *FormatFloatStr(double inNum);
extern swidget create_VectorOptions(swidget UxParent);
extern void JC_OverlayButton_CB(void);
extern void JC_PlotButton_CB(void);

/* globals */
extern JC_PlotOptions GLOBAL_PlotOptions; /* JC_ADDITION (12/11/95) */
swidget gSavedPlotOptions = NULL;
swidget PlotOptions;
extern swidget VectorOptions;
static int localPlotType;
Widget styleWidgets[7];
Widget thickWidgets[3];
Widget symbolWidgets[89];
extern Boolean gHiRez;


static	Widget	form18;
static	Widget	label_VectorOptions;
static	Widget	form_VectorOptions;
static	Widget	toggleButton_AspectCorrection;
static	Widget	label_Scale;
static	Widget	rowColumn_Scale;
static	Widget	toggleButton_ScaleAuto;
static	Widget	toggleButton_ScaleSameAsLast;
static	Widget	label_XSkip;
static	Widget	rowColumn_XSkip;
static	Widget	toggleButton_XSkipAuto;
static	Widget	toggleButton_XSkipCustom;
static	Widget	textField_XSkip;
static	Widget	label_YSkip;
static	Widget	rowColumn_YSkip;
static	Widget	toggleButton_YSkipAuto;
static	Widget	toggleButton_YSkipCustom;
static	Widget	textField_YSkip;
static	Widget	label_2DOptions;
static	Widget	form_2DOptions;
static	Widget	rowColumn_Levels;
static	Widget	toggleButton_LevelsAutoscale;
static	Widget	toggleButton_LevelsReuse;
static	Widget	toggleButton_LevelsCustom;
static	Widget	label_SCFLow;
static	Widget	label_SCFHigh;
static	Widget	label_SCFDelta;
static	Widget	toggleButton_SCFColorKey;
static	Widget	toggleButton_SCFOverlayContours;
static	Widget	label_1DOptions;
static	Widget	form_1DOptions;
static	Widget	toggleButton_LSAutoSymbols;
static	Widget	label_LSLineStyle;
static	Widget	label_LSSymbol;
static	Widget	lineStyle2;
static	Widget	lineStyle2_b8;
static	Widget	lineStyle1_b10;
static	Widget	lineStyle1_b11;
static	Widget	lineStyle1_b12;
static	Widget	lineStyle1_b13;
static	Widget	lineStyle1_b14;
static	Widget	lineStyle3;
static	Widget	lineStyle_b1;
static	Widget	optionMenu_p_b10;
static	Widget	lineStyle_b89;
static	Widget	lineStyle_b91;
static	Widget	lineStyle_b92;
static	Widget	lineStyle_b93;
static	Widget	lineStyle_b94;
static	Widget	lineStyle_b95;
static	Widget	lineStyle_b96;
static	Widget	lineStyle_b97;
static	Widget	lineStyle_b98;
static	Widget	lineStyle_b99;
static	Widget	lineStyle_b100;
static	Widget	lineStyle_b101;
static	Widget	lineStyle_b102;
static	Widget	lineStyle_b103;
static	Widget	lineStyle_b104;
static	Widget	lineStyle_b105;
static	Widget	lineStyle_b106;
static	Widget	lineStyle_b107;
static	Widget	lineStyle_b108;
static	Widget	lineStyle_b109;
static	Widget	lineStyle_b110;
static	Widget	lineStyle_b111;
static	Widget	lineStyle_b112;
static	Widget	lineStyle_b113;
static	Widget	lineStyle_b114;
static	Widget	lineStyle_b115;
static	Widget	lineStyle_b116;
static	Widget	lineStyle_b117;
static	Widget	lineStyle_b118;
static	Widget	lineStyle_b119;
static	Widget	lineStyle_b120;
static	Widget	lineStyle_b121;
static	Widget	lineStyle_b122;
static	Widget	lineStyle_b123;
static	Widget	lineStyle_b124;
static	Widget	lineStyle_b125;
static	Widget	lineStyle_b126;
static	Widget	lineStyle_b127;
static	Widget	lineStyle_b128;
static	Widget	lineStyle_b129;
static	Widget	lineStyle_b130;
static	Widget	lineStyle_b131;
static	Widget	lineStyle_b132;
static	Widget	lineStyle_b133;
static	Widget	lineStyle_b134;
static	Widget	lineStyle_b135;
static	Widget	lineStyle_b136;
static	Widget	lineStyle_b137;
static	Widget	lineStyle_b138;
static	Widget	lineStyle_b139;
static	Widget	lineStyle_b140;
static	Widget	lineStyle_b141;
static	Widget	lineStyle_b142;
static	Widget	lineStyle_b143;
static	Widget	lineStyle_b144;
static	Widget	lineStyle_b145;
static	Widget	lineStyle_b146;
static	Widget	lineStyle_b147;
static	Widget	lineStyle_b148;
static	Widget	lineStyle_b149;
static	Widget	lineStyle_b150;
static	Widget	lineStyle_b151;
static	Widget	lineStyle_b152;
static	Widget	lineStyle_b153;
static	Widget	lineStyle_b154;
static	Widget	lineStyle_b155;
static	Widget	lineStyle_b156;
static	Widget	lineStyle_b157;
static	Widget	lineStyle_b158;
static	Widget	lineStyle_b159;
static	Widget	lineStyle_b160;
static	Widget	lineStyle_b161;
static	Widget	lineStyle_b162;
static	Widget	lineStyle_b163;
static	Widget	lineStyle_b164;
static	Widget	lineStyle_b165;
static	Widget	lineStyle_b166;
static	Widget	lineStyle_b167;
static	Widget	lineStyle_b168;
static	Widget	lineStyle_b169;
static	Widget	lineStyle_b170;
static	Widget	lineStyle_b171;
static	Widget	lineStyle_b172;
static	Widget	lineStyle_b173;
static	Widget	lineStyle_b174;
static	Widget	lineStyle_b175;
static	Widget	lineStyle_b176;
static	Widget	label_LSThickness;
static	Widget	lineStyle1;
static	Widget	lineStyle2_b1;
static	Widget	lineStyle1_b2;
static	Widget	lineStyle4;
static	Widget	lineStyle4_b2;
static	Widget	lineStyle2_b2;
static	Widget	pushButton_Dismiss;
static	Widget	label_SelectOptions;
static	Widget	form_SelectOptions;
static	Widget	rowColumn_PlotType;
static	Widget	toggleButton_PlotTypeLineScatter;
static	Widget	toggleButton_PlotTypeShadeContourFill;
static	Widget	toggleButton_PlotTypeVector;
static	Widget	label_GeneralOptions;
static	Widget	form_GeneralOptions;
static	Widget	toggleButton_Transpose;
static	Widget	toggleButton_NoLabels;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "PlotOptions.h"
#undef CONTEXT_MACRO_ACCESS

Widget	PlotOptions;
Widget	frame_VectorOptions;
Widget	frame_2DOptions;
Widget	textField_SCFLow;
Widget	textField_SCFHigh;
Widget	textField_SCFDelta;
Widget	frame_1DOptions;
Widget	lineStyle1_b9;
Widget	optionMenu_LSLineStyle;
Widget	optionMenu_LSSymbol;
Widget	lineStyle1_b1;
Widget	optionMenu_LSThickness;
Widget	optionMenu18;
Widget	frame_SelectOptions;
Widget	frame_GeneralOptions;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static void DisableThickBtn()
{
  XtUnmanageChild(optionMenu_LSThickness);
  XtManageChild(optionMenu18);
}

static void EnableThickBtn()
{
  XtManageChild(optionMenu_LSThickness);
  XtUnmanageChild(optionMenu18);
}

void SetInitialPOState()
{
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);

	XmToggleButtonSetState(UxGetWidget(toggleButton_PlotTypeLineScatter), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton_PlotTypeShadeContourFill), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton_PlotTypeVector), False, False);

	XtUnmapWidget(UxGetWidget(frame_1DOptions));
	XtUnmapWidget(UxGetWidget(frame_VectorOptions));
	XtUnmapWidget(UxGetWidget(frame_2DOptions));
	switch (PO_ptr->plot_type) {
		case PLOT_LINE:
		case PLOT_SCATTER:
			XtUnmapWidget(UxGetWidget(frame_VectorOptions));
			XtUnmapWidget(UxGetWidget(frame_2DOptions));
			XtMapWidget(UxGetWidget(frame_1DOptions));
			XmToggleButtonSetState(UxGetWidget(toggleButton_PlotTypeLineScatter), True, False);
			XmToggleButtonSetState(UxGetWidget(toggleButton_LSAutoSymbols), True, False);
			localPlotType = PLOT_LINE;
			break;
		case PLOT_SHADE:
		case PLOT_CONTOUR:
		case PLOT_FILL:
			XtUnmapWidget(UxGetWidget(frame_1DOptions));
			XtUnmapWidget(UxGetWidget(frame_VectorOptions));
			XtMapWidget(UxGetWidget(frame_2DOptions));
			XmToggleButtonSetState(UxGetWidget(toggleButton_SCFColorKey), True, False);
			XmToggleButtonSetState(UxGetWidget(toggleButton_PlotTypeShadeContourFill), True, False);
			localPlotType = PLOT_SHADE;
			break;
		case PLOT_VECTOR:
			XtUnmapWidget(UxGetWidget(frame_1DOptions));
			XtUnmapWidget(UxGetWidget(frame_2DOptions));
			XtMapWidget(UxGetWidget(frame_VectorOptions));
			XmToggleButtonSetState(UxGetWidget(toggleButton_PlotTypeVector), True, False);
			localPlotType = PLOT_VECTOR;
			break;
			
	}
	PlotOptions2Interface();
}

UpdateLineStyleCB(wid, client_data, cbs)
Widget wid;
XtPointer client_data;
XtPointer cbs;
{
	char *tempText;
	XmString buttonLabel;
	int val=UNSET_VALUE, oldStyle=UNSET_VALUE;
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);
	
	/* option is encoded in button label */
	XtVaGetValues(wid,
		XmNlabelString, &buttonLabel,
		NULL);
	XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);

	if (tempText[0] == 'A') {
		/* auto selected--turn off the line thickness menu */
		PO_ptr->oneD_options.style = 0;
		DisableThickBtn();
	}
	else {
	        EnableThickBtn();
		sscanf(tempText, "%d", &val);
		oldStyle = PO_ptr->oneD_options.style;

		/* isolate just the line style */
		if ( PO_ptr->oneD_options.style > 6 && PO_ptr->oneD_options.style < 13)
			PO_ptr->oneD_options.style -= 6;
		else if (PO_ptr->oneD_options.style >= 13)
			PO_ptr->oneD_options.style -= 12;
		
		if ((val == 0 || val == 7 || val == 13) && PO_ptr->oneD_options.style > 0)
			/* selection from the line thickness menu */
			PO_ptr->oneD_options.style += val-1;
		else {
			/* selection from style menu--restore line thickness too (if any) */
			if (oldStyle > 6 && oldStyle < 13)
				/* double line */
				PO_ptr->oneD_options.style = val + 6;
			else if (oldStyle >= 13)
				/* triple line */
				PO_ptr->oneD_options.style = val + 12;
			else
				PO_ptr->oneD_options.style = val;
		}
	}
	XtFree(tempText); /* allocated with XmStringGetLtoR() */
}

UpdateLineSymbolCB(wid, client_data, cbs)
Widget wid;
XtPointer client_data;
XtPointer cbs;
{
	char *tempText;
	XmString buttonLabel;
	int val=UNSET_VALUE;
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);

	/* option is encoded in button label */
	XtVaGetValues(wid,
		XmNlabelString, &buttonLabel,
		NULL);
	XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);

	if (tempText[0] == 'A')
		PO_ptr->oneD_options.symbol = 0;
	else {
		sscanf(tempText, "%d", &val);
		PO_ptr->oneD_options.symbol = val;
	}
	XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


Update1DOptionsCB()
{
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_LSAutoSymbols)))
		PO_ptr->oneD_options.automatic = TRUE;
	else
		PO_ptr->oneD_options.automatic = FALSE;
}

Update2DOptionsCB()
{
	Boolean isSet;
	char *tText;
	float val;
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_LevelsAutoscale))) {
		/* autoscale levels */
		PO_ptr->twoD_options.level_type = NO_LEVELS;
                XtVaSetValues(UxGetWidget(textField_SCFLow), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField_SCFHigh), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField_SCFDelta), 
			XmNvalue, "",
			NULL);
	}
	else if (XmToggleButtonGetState(UxGetWidget(toggleButton_LevelsReuse))) {
		/* Reuse last */
		PO_ptr->twoD_options.level_type = LAST_LEVELS;
                XtVaSetValues(UxGetWidget(textField_SCFLow), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField_SCFHigh), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField_SCFDelta), 
			XmNvalue, "",
			NULL);
	}
	else if (XmToggleButtonGetState(UxGetWidget(toggleButton_LevelsCustom))) {
		/* custom levels */
		PO_ptr->twoD_options.level_type = NEW_LEVELS;

		/* get the value of low */
		tText = XmTextFieldGetString(UxGetWidget(textField_SCFLow));
		if (strlen(tText) == 0)
			PO_ptr->twoD_options.levels[LO] = UNSET_VALUE;
		else {
			sscanf(tText, "%f", &val);
			PO_ptr->twoD_options.levels[LO] = val;
		}

		/* get the value of high */
		tText = XmTextFieldGetString(UxGetWidget(textField_SCFHigh));
		if (strlen(tText) == 0)
			PO_ptr->twoD_options.levels[HI] = UNSET_VALUE;
		else {
			sscanf(tText, "%f", &val);
			PO_ptr->twoD_options.levels[HI] = val;
		}

		/* get the value of delta */
		tText = XmTextFieldGetString(UxGetWidget(textField_SCFDelta));
		if (strlen(tText) == 0)
			PO_ptr->twoD_options.levels[DELTA] = UNSET_VALUE;
		else {
			sscanf(tText, "%f", &val);
			PO_ptr->twoD_options.levels[DELTA] = val;
		}

	    XtFree(tText); /* allocated with XmTextFieldGetString() */
	}

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_SCFColorKey))) {
		PO_ptr->twoD_options.key = TRUE;
		PO_ptr->twoD_options.no_key = FALSE;
	 } else {
		PO_ptr->twoD_options.key = FALSE;
		PO_ptr->twoD_options.no_key = TRUE;
	}

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_SCFOverlayContours)) &&
		XtIsSensitive(UxGetWidget(toggleButton_SCFOverlayContours)))
		/* overlay contour lines */
		PO_ptr->twoD_options.line = TRUE;
	else 
		PO_ptr->twoD_options.line = FALSE;

}

UpdateVectorOptionsCB()
{
	char *tText;
	int val;
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_AspectCorrection)))
		/* use aspect correction */
		PO_ptr->vector_options.aspect = TRUE;
	else 
		PO_ptr->vector_options.aspect = FALSE;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_ScaleAuto)))
		/* autoscale levels */
		PO_ptr->vector_options.length_type = NO_LENGTH;
	else if (XmToggleButtonGetState(UxGetWidget(toggleButton_ScaleSameAsLast)))
		/* Reuse last length */
		PO_ptr->vector_options.length_type = LAST_LENGTH;
	else
		/* custom length */
		 PO_ptr->vector_options.length_type = NEW_LENGTH;

	/* get the value of y skip */
	tText = XmTextFieldGetString(UxGetWidget(textField_XSkip));
	if (strlen(tText) == 0)
	        PO_ptr->vector_options.xskip = 1;
	else {
		sscanf(tText, "%d", &val);
		PO_ptr->vector_options.xskip = val;
	}

	/* get the value of y skip */
	tText = XmTextFieldGetString(UxGetWidget(textField_YSkip));
	if (strlen(tText) == 0)
	        PO_ptr->vector_options.yskip = 1;
	else {
		sscanf(tText, "%d", &val);
		PO_ptr->vector_options.yskip = val;
	}

	XtFree(tText); /* allocated with XmTextFieldGetSTring() */
}

void UpdateGeneralOptionsCB(void)
{
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_Transpose)))
		PO_ptr->transpose = TRUE;
	else 
		PO_ptr->transpose = FALSE;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton_NoLabels)))
		PO_ptr->nolabels = TRUE;
	else 
		PO_ptr->nolabels = FALSE;
}


void PlotOptions2Interface()
{
	Widget wid;
	char tText[32];
	JC_PlotOptions *PO_ptr=&(GLOBAL_PlotOptions);

	/* Make interface reflect state of plot options */

	/* 1D */
	/* set the line style option menu */
	if (PO_ptr->oneD_options.style <= 6)
		wid = styleWidgets[PO_ptr->oneD_options.style];
	else if (PO_ptr->oneD_options.style <= 12)
		wid = styleWidgets[PO_ptr->oneD_options.style-6];
	else if (PO_ptr->oneD_options.style <= 18)
		wid = styleWidgets[PO_ptr->oneD_options.style-12];

	XtVaSetValues(UxGetWidget(optionMenu_LSLineStyle),
		XmNmenuHistory, wid,
		NULL);
	XtUnmapWidget(UxGetWidget(optionMenu_LSLineStyle));
	XtMapWidget(UxGetWidget(optionMenu_LSLineStyle));

	/* line thickness option menu */
	if (PO_ptr->oneD_options.style <= 6)
		wid = thickWidgets[0];
	else if (PO_ptr->oneD_options.style <= 12)
		wid = thickWidgets[1];
	else if (PO_ptr->oneD_options.style <= 18)
		wid = thickWidgets[2];

	XtVaSetValues(UxGetWidget(optionMenu_LSThickness),
		XmNmenuHistory, wid,
		NULL);
	XtUnmapWidget(UxGetWidget(optionMenu_LSThickness));
	XtMapWidget(UxGetWidget(optionMenu_LSThickness));

	if (PO_ptr->oneD_options.style)
	        DisableThickBtn();
	else
	        EnableThickBtn();

	/* symbol option menu */
	wid = symbolWidgets[PO_ptr->oneD_options.symbol];
	XtVaSetValues(UxGetWidget(optionMenu_LSSymbol),
		XmNmenuHistory, wid,
		NULL);
	XtUnmapWidget(UxGetWidget(optionMenu_LSSymbol));
	XtMapWidget(UxGetWidget(optionMenu_LSSymbol));

	if (PO_ptr->oneD_options.automatic) {
		XmToggleButtonSetState(UxGetWidget(toggleButton_LSAutoSymbols), True, False);
		XtSetSensitive(UxGetWidget(optionMenu_LSLineStyle), False);
		XtSetSensitive(UxGetWidget(optionMenu_LSSymbol), False);
	        DisableThickBtn();
	}
	else {
		XmToggleButtonSetState(UxGetWidget(toggleButton_LSAutoSymbols), False, False);
		XtSetSensitive(UxGetWidget(optionMenu_LSLineStyle), True);
		XtSetSensitive(UxGetWidget(optionMenu_LSSymbol), True);
	        EnableThickBtn();
	}

	/* 2D */
	XmToggleButtonSetState(UxGetWidget(toggleButton_LevelsAutoscale), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton_LevelsReuse), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton_LevelsCustom), False, False);
        XtVaSetValues(UxGetWidget(textField_SCFLow), 
		XmNvalue, "",
		NULL);
       	XtVaSetValues(UxGetWidget(textField_SCFHigh), 
		XmNvalue, "",
		NULL);
	XtVaSetValues(UxGetWidget(textField_SCFDelta), 
		XmNvalue, "",
		NULL);

	if (PO_ptr->twoD_options.level_type == NO_LEVELS) {
		/* autoscale levels */
		XmToggleButtonSetState(UxGetWidget(toggleButton_LevelsAutoscale), True, False);
		XtSetSensitive(UxGetWidget(textField_SCFLow), False);
		XtSetSensitive(UxGetWidget(textField_SCFHigh), False);
		XtSetSensitive(UxGetWidget(textField_SCFDelta), False);
		XtSetSensitive(UxGetWidget(label_SCFLow), False);
		XtSetSensitive(UxGetWidget(label_SCFHigh), False);
		XtSetSensitive(UxGetWidget(label_SCFDelta), False);
	}
	else if (PO_ptr->twoD_options.level_type == LAST_LEVELS) {
		/* Reuse last */
		XmToggleButtonSetState(UxGetWidget(toggleButton_LevelsReuse), True, False);
		XtSetSensitive(UxGetWidget(textField_SCFLow), False);
		XtSetSensitive(UxGetWidget(textField_SCFHigh), False);
		XtSetSensitive(UxGetWidget(textField_SCFDelta), False);
		XtSetSensitive(UxGetWidget(label_SCFLow), False);
		XtSetSensitive(UxGetWidget(label_SCFHigh), False);
		XtSetSensitive(UxGetWidget(label_SCFDelta), False);
	}
	else if (PO_ptr->twoD_options.level_type == NEW_LEVELS) {
		/* custom levels */
		XtSetSensitive(UxGetWidget(textField_SCFLow), True);
		XtSetSensitive(UxGetWidget(textField_SCFHigh), True);
		XtSetSensitive(UxGetWidget(textField_SCFDelta), True);
		XtSetSensitive(UxGetWidget(label_SCFLow), True);
		XtSetSensitive(UxGetWidget(label_SCFHigh), True);
		XtSetSensitive(UxGetWidget(label_SCFDelta), True);

		XmToggleButtonSetState(UxGetWidget(toggleButton_LevelsCustom), True, False);

		/* set the value of low */
		sprintf(tText, "%f", PO_ptr->twoD_options.levels[LO]);
		XmTextFieldSetString(UxGetWidget(textField_SCFLow), tText);

		/* set the value of high */
		sprintf(tText, "%f", PO_ptr->twoD_options.levels[HI]);
		XmTextFieldSetString(UxGetWidget(textField_SCFHigh), tText);

		/* set the value of delta */
		sprintf(tText, "%f", PO_ptr->twoD_options.levels[DELTA]);
		XmTextFieldSetString(UxGetWidget(textField_SCFDelta), tText);
	}

	if ( PO_ptr->twoD_options.key )
		/* use a color key */
		XmToggleButtonSetState(UxGetWidget(toggleButton_SCFColorKey), True, False);
	else 
		XmToggleButtonSetState(UxGetWidget(toggleButton_SCFColorKey), False, False);

	if ( PO_ptr->twoD_options.line )
		/* overlay contour lines */
		XmToggleButtonSetState(UxGetWidget(toggleButton_SCFOverlayContours), True, False);
	else 
		XmToggleButtonSetState(UxGetWidget(toggleButton_SCFOverlayContours), False, False);

	/* Vector */	
	if ( PO_ptr->vector_options.aspect )
		/* use aspect correction */
		XmToggleButtonSetState(UxGetWidget(toggleButton_AspectCorrection), True, False);
	else 
		XmToggleButtonSetState(UxGetWidget(toggleButton_AspectCorrection), False, False);

	XmToggleButtonSetState(UxGetWidget(toggleButton_ScaleAuto), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton_ScaleSameAsLast), False, False);
	if ( PO_ptr->vector_options.length_type == NO_LENGTH )
		/* autoscale levels */
		XmToggleButtonSetState(UxGetWidget(toggleButton_ScaleAuto), True, False);
	else if (PO_ptr->vector_options.length_type == LAST_LENGTH )
		/* Reuse last length */
		XmToggleButtonSetState(UxGetWidget(toggleButton_ScaleSameAsLast), True, False);


	XmToggleButtonSetState(UxGetWidget(toggleButton_XSkipAuto), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton_XSkipCustom), False, False);
	if ( PO_ptr->vector_options.xskip == 0 ) {
		/* auto x skip */
		XmToggleButtonSetState(UxGetWidget(toggleButton_XSkipAuto), True, False);
		XtUnmapWidget(UxGetWidget(textField_XSkip));
	}
	else {
		XmToggleButtonSetState(UxGetWidget(toggleButton_XSkipCustom), True, False);
		/* set the value of x skip */
		sprintf(tText, "%d", PO_ptr->vector_options.xskip);
		XmTextFieldSetString(UxGetWidget(textField_XSkip), tText);
		XtMapWidget(UxGetWidget(textField_XSkip));
	}

	XmToggleButtonSetState(UxGetWidget(toggleButton_YSkipAuto), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton_YSkipCustom), False, False);
	if ( PO_ptr->vector_options.yskip == 0)
		/* auto y skip */ {
		XmToggleButtonSetState(UxGetWidget(toggleButton_YSkipAuto), True, False);
		XtUnmapWidget(UxGetWidget(textField_YSkip));
	}
	else {
		XmToggleButtonSetState(UxGetWidget(toggleButton_YSkipCustom), True, False);
		/* set the value of y skip */
		sprintf(tText, "%d", PO_ptr->vector_options.yskip);
		XmTextFieldSetString(UxGetWidget(textField_YSkip), tText);
		XtMapWidget(UxGetWidget(textField_YSkip));
	}

	if ( PO_ptr->transpose )
		XmToggleButtonSetState(UxGetWidget(toggleButton_Transpose), TRUE, FALSE);
	else
		XmToggleButtonSetState(UxGetWidget(toggleButton_Transpose), FALSE, FALSE);

	if ( PO_ptr->nolabels )
		XmToggleButtonSetState(UxGetWidget(toggleButton_NoLabels), TRUE, FALSE);
	else
		XmToggleButtonSetState(UxGetWidget(toggleButton_NoLabels), FALSE, FALSE);

}

static void InitArrays()
{

	styleWidgets[0] = UxGetWidget(lineStyle2_b8);
	styleWidgets[1] = UxGetWidget(lineStyle1_b9);
	styleWidgets[2] = UxGetWidget(lineStyle1_b10);
	styleWidgets[3] = UxGetWidget(lineStyle1_b11);
	styleWidgets[4] = UxGetWidget(lineStyle1_b12);
	styleWidgets[5] = UxGetWidget(lineStyle1_b13);
	styleWidgets[6] = UxGetWidget(lineStyle1_b14); 

	thickWidgets[0] = UxGetWidget(lineStyle2_b1);
	thickWidgets[1] = UxGetWidget(lineStyle1_b1);
	thickWidgets[2] = UxGetWidget(lineStyle1_b2);

	symbolWidgets[0] = UxGetWidget(lineStyle_b1);
	symbolWidgets[1] = UxGetWidget(optionMenu_p_b10);
	symbolWidgets[2] = UxGetWidget(lineStyle_b89);
	symbolWidgets[3] = UxGetWidget(lineStyle_b91);
	symbolWidgets[4] = UxGetWidget(lineStyle_b92);
	symbolWidgets[5] = UxGetWidget(lineStyle_b93);
	symbolWidgets[6] = UxGetWidget(lineStyle_b94);
	symbolWidgets[7] = UxGetWidget(lineStyle_b95);
	symbolWidgets[8] = UxGetWidget(lineStyle_b96);
	symbolWidgets[9] = UxGetWidget(lineStyle_b97);
	symbolWidgets[10] = UxGetWidget(lineStyle_b98);
	symbolWidgets[11] = UxGetWidget(lineStyle_b99);
	symbolWidgets[12] = UxGetWidget(lineStyle_b100);
	symbolWidgets[13] = UxGetWidget(lineStyle_b101);
	symbolWidgets[14] = UxGetWidget(lineStyle_b102);
	symbolWidgets[15] = UxGetWidget(lineStyle_b103);
	symbolWidgets[16] = UxGetWidget(lineStyle_b104);
	symbolWidgets[17] = UxGetWidget(lineStyle_b105);
	symbolWidgets[18] = UxGetWidget(lineStyle_b106);
	symbolWidgets[19] = UxGetWidget(lineStyle_b107);
	symbolWidgets[20] = UxGetWidget(lineStyle_b108);
	symbolWidgets[21] = UxGetWidget(lineStyle_b109);
	symbolWidgets[22] = UxGetWidget(lineStyle_b110);
	symbolWidgets[23] = UxGetWidget(lineStyle_b111);
	symbolWidgets[24] = UxGetWidget(lineStyle_b112);
	symbolWidgets[25] = UxGetWidget(lineStyle_b113);
	symbolWidgets[26] = UxGetWidget(lineStyle_b114);
	symbolWidgets[27] = UxGetWidget(lineStyle_b115);
	symbolWidgets[28] = UxGetWidget(lineStyle_b116);
	symbolWidgets[29] = UxGetWidget(lineStyle_b117);
	symbolWidgets[30] = UxGetWidget(lineStyle_b118);
	symbolWidgets[31] = UxGetWidget(lineStyle_b119);
	symbolWidgets[32] = UxGetWidget(lineStyle_b120);
	symbolWidgets[33] = UxGetWidget(lineStyle_b121);
	symbolWidgets[34] = UxGetWidget(lineStyle_b122);
	symbolWidgets[35] = UxGetWidget(lineStyle_b123);
	symbolWidgets[36] = UxGetWidget(lineStyle_b124);
	symbolWidgets[37] = UxGetWidget(lineStyle_b125);
	symbolWidgets[38] = UxGetWidget(lineStyle_b126);
	symbolWidgets[39] = UxGetWidget(lineStyle_b127);
	symbolWidgets[40] = UxGetWidget(lineStyle_b128);
	symbolWidgets[41] = UxGetWidget(lineStyle_b129);
	symbolWidgets[42] = UxGetWidget(lineStyle_b130);
	symbolWidgets[43] = UxGetWidget(lineStyle_b131);
	symbolWidgets[44] = UxGetWidget(lineStyle_b132);
	symbolWidgets[45] = UxGetWidget(lineStyle_b133);
	symbolWidgets[46] = UxGetWidget(lineStyle_b134);
	symbolWidgets[47] = UxGetWidget(lineStyle_b135);
	symbolWidgets[48] = UxGetWidget(lineStyle_b136);
	symbolWidgets[49] = UxGetWidget(lineStyle_b137);
	symbolWidgets[50] = UxGetWidget(lineStyle_b138);
	symbolWidgets[51] = UxGetWidget(lineStyle_b139);
	symbolWidgets[52] = UxGetWidget(lineStyle_b140);
	symbolWidgets[53] = UxGetWidget(lineStyle_b141);
	symbolWidgets[54] = UxGetWidget(lineStyle_b142);
	symbolWidgets[55] = UxGetWidget(lineStyle_b143);
	symbolWidgets[56] = UxGetWidget(lineStyle_b144);
	symbolWidgets[57] = UxGetWidget(lineStyle_b145);
	symbolWidgets[58] = UxGetWidget(lineStyle_b146);
	symbolWidgets[59] = UxGetWidget(lineStyle_b147);
	symbolWidgets[60] = UxGetWidget(lineStyle_b148);
	symbolWidgets[61] = UxGetWidget(lineStyle_b149);
	symbolWidgets[62] = UxGetWidget(lineStyle_b150);
	symbolWidgets[63] = UxGetWidget(lineStyle_b151);
	symbolWidgets[64] = UxGetWidget(lineStyle_b152);
	symbolWidgets[65] = UxGetWidget(lineStyle_b153);
	symbolWidgets[66] = UxGetWidget(lineStyle_b154);
	symbolWidgets[67] = UxGetWidget(lineStyle_b155);
	symbolWidgets[68] = UxGetWidget(lineStyle_b156);
	symbolWidgets[69] = UxGetWidget(lineStyle_b157);
	symbolWidgets[70] = UxGetWidget(lineStyle_b158);
	symbolWidgets[71] = UxGetWidget(lineStyle_b159);
	symbolWidgets[72] = UxGetWidget(lineStyle_b160);
	symbolWidgets[73] = UxGetWidget(lineStyle_b161);
	symbolWidgets[74] = UxGetWidget(lineStyle_b162);
	symbolWidgets[75] = UxGetWidget(lineStyle_b163);
	symbolWidgets[76] = UxGetWidget(lineStyle_b164);
	symbolWidgets[77] = UxGetWidget(lineStyle_b165);
	symbolWidgets[78] = UxGetWidget(lineStyle_b166);
	symbolWidgets[79] = UxGetWidget(lineStyle_b167);
	symbolWidgets[80] = UxGetWidget(lineStyle_b168);
	symbolWidgets[81] = UxGetWidget(lineStyle_b169);
	symbolWidgets[82] = UxGetWidget(lineStyle_b170);
	symbolWidgets[83] = UxGetWidget(lineStyle_b171);
	symbolWidgets[84] = UxGetWidget(lineStyle_b172);
	symbolWidgets[85] = UxGetWidget(lineStyle_b173);
	symbolWidgets[86] = UxGetWidget(lineStyle_b174);
	symbolWidgets[87] = UxGetWidget(lineStyle_b175);
	symbolWidgets[88] = UxGetWidget(lineStyle_b176);
}

static void InitPixmaps()
{
	XtVaSetValues(UxGetWidget(styleWidgets[1]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_1_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_1_ins_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[2]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_2_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[3]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_3_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[4]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_4_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[5]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_5_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[6]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_6_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(thickWidgets[0]),
		XmNlabelType, XmPIXMAP,
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_single_ins_xpm),
		XmNlabelPixmap, GetPixmapFromData(lines_single_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(thickWidgets[1]),
		XmNlabelType, XmPIXMAP,
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_double_ins_xpm),
		XmNlabelPixmap, GetPixmapFromData(lines_double_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(thickWidgets[2]),
		XmNlabelType, XmPIXMAP,
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_triple_ins_xpm),
		XmNlabelPixmap, GetPixmapFromData(lines_triple_xpm),
		NULL);

#ifdef FULL_GUI_VERSION
	XtVaSetValues(UxGetWidget(symbolWidgets[1]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_1_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[2]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_2_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[3]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_3_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[4]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_4_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[5]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_5_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[6]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_6_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[7]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_7_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[8]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_8_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[9]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_9_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[10]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_10_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[11]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_11_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[12]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_12_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[13]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_13_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[14]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_14_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[15]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_15_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[16]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_16_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[17]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_17_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[18]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_18_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[19]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_19_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[20]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_20_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[21]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_21_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[22]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_22_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[23]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_23_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[24]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_24_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[25]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_25_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[26]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_26_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[27]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_27_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[28]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_28_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[29]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_29_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[30]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_30_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[31]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_31_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[32]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_32_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[33]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_33_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[34]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_34_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[35]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_35_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[36]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_36_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[37]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_37_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[38]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_38_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[39]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_39_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[40]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_40_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[41]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_41_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[42]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_42_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[43]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_43_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[44]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_44_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[45]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_45_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[46]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_46_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[47]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_47_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[48]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_48_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[49]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_49_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[50]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_50_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[51]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_51_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[52]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_52_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[53]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_53_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[54]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_54_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[55]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_55_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[56]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_56_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[57]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_57_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[58]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_58_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[59]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_59_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[60]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_60_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[61]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_61_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[62]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_62_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[63]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_63_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[64]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_64_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[65]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_65_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[66]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_66_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[67]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_67_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[68]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_68_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[69]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_69_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[70]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_70_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[71]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_71_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[72]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_72_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[73]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_73_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[74]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_74_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[75]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_75_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[76]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_76_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[77]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_77_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[78]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_78_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[79]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_79_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[80]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_80_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[81]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_81_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[82]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_82_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[83]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_83_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[84]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_84_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[85]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_85_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[86]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_86_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[87]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_87_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[88]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_88_xpm),
		NULL);
#endif
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	destroyCB_PlotOptions(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedPlotOptions = NULL;
}

static	void	valueChangedCB_toggleButton_AspectCorrection(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_toggleButton_ScaleAuto(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_toggleButton_ScaleSameAsLast(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_toggleButton_XSkipAuto(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_toggleButton_XSkipCustom(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set == True)
		XtMapWidget(UxGetWidget(textField_XSkip));
	else
		XtUnmapWidget(UxGetWidget(textField_XSkip));
	UpdateVectorOptionsCB();
	}
}

static	void	activateCB_textField_XSkip(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	losingFocusCB_textField_XSkip(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_textField_XSkip(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_toggleButton_YSkipAuto(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_toggleButton_YSkipCustom(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set == True)
		XtMapWidget(UxGetWidget(textField_YSkip));
	else
		XtUnmapWidget(UxGetWidget(textField_YSkip));
	UpdateVectorOptionsCB();
	}
}

static	void	activateCB_textField_YSkip(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	losingFocusCB_textField_YSkip(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_textField_YSkip(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateVectorOptionsCB();
}

static	void	valueChangedCB_toggleButton_LevelsAutoscale(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	valueChangedCB_toggleButton_LevelsReuse(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	valueChangedCB_toggleButton_LevelsCustom(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget CustomLevels;
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	/*if (cbInfo->set == True)
		CustomLevels = create_CustomLevels(NO_PARENT);
	else
		XtPopdown(UxGetWidget(CustomLevels)); */
	
	if (cbInfo->set == True) {
		XtSetSensitive(UxGetWidget(textField_SCFLow), True);
		XtSetSensitive(UxGetWidget(textField_SCFHigh), True);
		XtSetSensitive(UxGetWidget(textField_SCFDelta), True);
		XtSetSensitive(UxGetWidget(label_SCFLow), True);
		XtSetSensitive(UxGetWidget(label_SCFHigh), True);
		XtSetSensitive(UxGetWidget(label_SCFDelta), True);
	}
	else {
		XtSetSensitive(UxGetWidget(textField_SCFLow), False);
		XtSetSensitive(UxGetWidget(textField_SCFHigh), False);
		XtSetSensitive(UxGetWidget(textField_SCFDelta), False);
		XtSetSensitive(UxGetWidget(label_SCFLow), False);
		XtSetSensitive(UxGetWidget(label_SCFHigh), False);
		XtSetSensitive(UxGetWidget(label_SCFDelta), False);
	}
	Update2DOptionsCB();
	
	}
}

static	void	activateCB_textField_SCFLow(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	losingFocusCB_textField_SCFLow(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	activateCB_textField_SCFHigh(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	losingFocusCB_textField_SCFHigh(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	activateCB_textField_SCFDelta(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	losingFocusCB_textField_SCFDelta(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	valueChangedCB_toggleButton_SCFColorKey(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	valueChangedCB_toggleButton_SCFOverlayContours(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Update2DOptionsCB();
}

static	void	valueChangedCB_toggleButton_LSAutoSymbols(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set == True) {
		/* desensitize the line symbol and line width option menus */
		XtSetSensitive(UxGetWidget(optionMenu_LSLineStyle), False);
		XtSetSensitive(UxGetWidget(optionMenu_LSSymbol), False);
		XtSetSensitive(UxGetWidget(optionMenu_LSThickness), False);
	}
	else {
		/* sensitize the line symbol and line width option menus */
		XtSetSensitive(UxGetWidget(optionMenu_LSLineStyle), True);
		XtSetSensitive(UxGetWidget(optionMenu_LSSymbol), True);
		XtSetSensitive(UxGetWidget(optionMenu_LSThickness), True);
	}
	Update1DOptionsCB();
	}
}

static	void	activateCB_lineStyle2_b8(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b10(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b11(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b12(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b13(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b14(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p_b10(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b89(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b91(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b92(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b93(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b94(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b95(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b96(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b97(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b98(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b99(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b100(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b101(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b102(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b103(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b104(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b105(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b106(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b107(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b108(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b109(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b110(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b111(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b112(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b113(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b114(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b115(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b116(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b117(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b118(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b119(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b120(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b121(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b122(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b123(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b124(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b125(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b126(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b127(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b128(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b129(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b130(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b131(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b132(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b133(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b134(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b135(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b136(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b137(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b138(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b139(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b140(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b141(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b142(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b143(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b144(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b145(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b146(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b147(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b148(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b149(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b150(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b151(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b152(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b153(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b154(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b155(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b156(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b157(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b158(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b159(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b160(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b161(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b162(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b163(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b164(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b165(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b166(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b167(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b168(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b169(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b170(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b171(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b172(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b173(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b174(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b175(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle_b176(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineSymbolCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle2_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle1_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_lineStyle2_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	UpdateLineStyleCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_pushButton_Dismiss(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XtPopdown(UxGetWidget(PlotOptions));
	
	}
}

static	void	valueChangedCB_toggleButton_PlotTypeLineScatter(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True) {
		localPlotType = PLOT_LINE;
		PlotOptions2Interface();
		XtUnmapWidget(UxGetWidget(frame_2DOptions));
		XtUnmapWidget(UxGetWidget(frame_VectorOptions));
		XtMapWidget(UxGetWidget(frame_1DOptions));	
	}
	}
}

static	void	valueChangedCB_toggleButton_PlotTypeShadeContourFill(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True) {
		localPlotType = PLOT_SHADE;
		PlotOptions2Interface();
		XtUnmapWidget(UxGetWidget(frame_1DOptions));
		XtUnmapWidget(UxGetWidget(frame_VectorOptions));
		XtMapWidget(UxGetWidget(frame_2DOptions));
	}
	
	}
}

static	void	valueChangedCB_toggleButton_PlotTypeVector(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XmToggleButtonCallbackStruct *callBackInfo = (XmToggleButtonCallbackStruct *) UxCallbackArg;
	
	if (callBackInfo->set == True) {
		localPlotType = PLOT_VECTOR;
		PlotOptions2Interface();
		XtUnmapWidget(UxGetWidget(frame_1DOptions));
		XtUnmapWidget(UxGetWidget(frame_2DOptions));
		XtMapWidget(UxGetWidget(frame_VectorOptions));
	}
	}
}

static	void	valueChangedCB_toggleButton_Transpose(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateGeneralOptionsCB();
}

static	void	valueChangedCB_toggleButton_NoLabels(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	UpdateGeneralOptionsCB();
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_PlotOptions()
{
	Widget		_UxParent;
	Widget		lineStyle2_shell;
	Widget		lineStyle3_shell;
	Widget		lineStyle1_shell;
	Widget		lineStyle4_shell;


	/* Creation of PlotOptions */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	PlotOptions = XtVaCreatePopupShell( "PlotOptions",
			topLevelShellWidgetClass,
			_UxParent,
			XmNx, 50,
			XmNy, 191,
			XmNiconName, "Ferret: Plot Options",
			XmNtitle, "Ferret Plot Options",
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( PlotOptions, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_PlotOptions,
		(XtPointer) NULL );



	/* Creation of form18 */
	form18 = XtVaCreateManagedWidget( "form18",
			xmFormWidgetClass,
			PlotOptions,
			XmNresizePolicy, XmRESIZE_ANY,
			XmNx, 50,
			XmNy, 36,
			XmNnoResize, TRUE,
			NULL );


	/* Creation of frame_VectorOptions */
	frame_VectorOptions = XtVaCreateManagedWidget( "frame_VectorOptions",
			xmFrameWidgetClass,
			form18,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNheight, 195,
			NULL );


	/* Creation of label_VectorOptions */
	label_VectorOptions = XtVaCreateManagedWidget( "label_VectorOptions",
			xmLabelWidgetClass,
			frame_VectorOptions,
			RES_CONVERT( XmNlabelString, "Vector Options" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_VectorOptions */
	form_VectorOptions = XtVaCreateManagedWidget( "form_VectorOptions",
			xmFormWidgetClass,
			frame_VectorOptions,
			XmNresizePolicy, XmRESIZE_NONE,
			NULL );


	/* Creation of toggleButton_AspectCorrection */
	toggleButton_AspectCorrection = XtVaCreateManagedWidget( "toggleButton_AspectCorrection",
			xmToggleButtonWidgetClass,
			form_VectorOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Aspect Correction" ),
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 0,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );
	XtAddCallback( toggleButton_AspectCorrection, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_AspectCorrection,
		(XtPointer) NULL );



	/* Creation of label_Scale */
	label_Scale = XtVaCreateManagedWidget( "label_Scale",
			xmLabelWidgetClass,
			form_VectorOptions,
			XmNalignment, XmALIGNMENT_END,
			RES_CONVERT( XmNlabelString, "Scale:" ),
			XmNleftOffset, 22,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNtopWidget, toggleButton_AspectCorrection,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of rowColumn_Scale */
	rowColumn_Scale = XtVaCreateManagedWidget( "rowColumn_Scale",
			xmRowColumnWidgetClass,
			form_VectorOptions,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNentryVerticalAlignment, XmALIGNMENT_CENTER,
			XmNmarginHeight, 3,
			XmNmarginWidth, 0,
			XmNpacking, XmPACK_TIGHT,
			XmNspacing, 3,
			XmNleftOffset, 4,
			XmNleftWidget, label_Scale,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 2,
			XmNtopWidget, toggleButton_AspectCorrection,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of toggleButton_ScaleAuto */
	toggleButton_ScaleAuto = XtVaCreateManagedWidget( "toggleButton_ScaleAuto",
			xmToggleButtonWidgetClass,
			rowColumn_Scale,
			RES_CONVERT( XmNlabelString, "Auto" ),
			NULL );
	XtAddCallback( toggleButton_ScaleAuto, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_ScaleAuto,
		(XtPointer) NULL );



	/* Creation of toggleButton_ScaleSameAsLast */
	toggleButton_ScaleSameAsLast = XtVaCreateManagedWidget( "toggleButton_ScaleSameAsLast",
			xmToggleButtonWidgetClass,
			rowColumn_Scale,
			RES_CONVERT( XmNlabelString, "Same As Last" ),
			NULL );
	XtAddCallback( toggleButton_ScaleSameAsLast, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_ScaleSameAsLast,
		(XtPointer) NULL );



	/* Creation of label_XSkip */
	label_XSkip = XtVaCreateManagedWidget( "label_XSkip",
			xmLabelWidgetClass,
			form_VectorOptions,
			XmNalignment, XmALIGNMENT_END,
			RES_CONVERT( XmNlabelString, "X-Skip:" ),
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 14,
			XmNtopWidget, label_Scale,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of rowColumn_XSkip */
	rowColumn_XSkip = XtVaCreateManagedWidget( "rowColumn_XSkip",
			xmRowColumnWidgetClass,
			form_VectorOptions,
			XmNx, 81,
			XmNy, 69,
			XmNwidth, 124,
			XmNheight, 21,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNpacking, XmPACK_TIGHT,
			XmNleftOffset, 0,
			XmNleftWidget, label_XSkip,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 1,
			XmNtopWidget, rowColumn_Scale,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of toggleButton_XSkipAuto */
	toggleButton_XSkipAuto = XtVaCreateManagedWidget( "toggleButton_XSkipAuto",
			xmToggleButtonWidgetClass,
			rowColumn_XSkip,
			RES_CONVERT( XmNlabelString, "Auto" ),
			XmNy, 2,
			NULL );
	XtAddCallback( toggleButton_XSkipAuto, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_XSkipAuto,
		(XtPointer) NULL );



	/* Creation of toggleButton_XSkipCustom */
	toggleButton_XSkipCustom = XtVaCreateManagedWidget( "toggleButton_XSkipCustom",
			xmToggleButtonWidgetClass,
			rowColumn_XSkip,
			RES_CONVERT( XmNlabelString, "Custom" ),
			NULL );
	XtAddCallback( toggleButton_XSkipCustom, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_XSkipCustom,
		(XtPointer) NULL );



	/* Creation of textField_XSkip */
	textField_XSkip = XtVaCreateManagedWidget( "textField_XSkip",
			xmTextFieldWidgetClass,
			form_VectorOptions,
			XmNsensitive, TRUE,
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 18,
			XmNleftWidget, rowColumn_XSkip,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 1,
			XmNtopWidget, rowColumn_Scale,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNwidth, 100,
			NULL );
	XtAddCallback( textField_XSkip, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_XSkip,
		(XtPointer) NULL );
	XtAddCallback( textField_XSkip, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_XSkip,
		(XtPointer) NULL );
	XtAddCallback( textField_XSkip, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField_XSkip,
		(XtPointer) NULL );



	/* Creation of label_YSkip */
	label_YSkip = XtVaCreateManagedWidget( "label_YSkip",
			xmLabelWidgetClass,
			form_VectorOptions,
			XmNalignment, XmALIGNMENT_END,
			RES_CONVERT( XmNlabelString, "Y-Skip:" ),
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 20,
			XmNtopWidget, label_XSkip,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNy, 111,
			XmNheight, 14,
			NULL );


	/* Creation of rowColumn_YSkip */
	rowColumn_YSkip = XtVaCreateManagedWidget( "rowColumn_YSkip",
			xmRowColumnWidgetClass,
			form_VectorOptions,
			XmNx, 82,
			XmNy, 104,
			XmNwidth, 124,
			XmNheight, 21,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNpacking, XmPACK_TIGHT,
			XmNleftOffset, 0,
			XmNleftWidget, label_YSkip,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 5,
			XmNtopWidget, rowColumn_XSkip,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of toggleButton_YSkipAuto */
	toggleButton_YSkipAuto = XtVaCreateManagedWidget( "toggleButton_YSkipAuto",
			xmToggleButtonWidgetClass,
			rowColumn_YSkip,
			RES_CONVERT( XmNlabelString, "Auto" ),
			NULL );
	XtAddCallback( toggleButton_YSkipAuto, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_YSkipAuto,
		(XtPointer) NULL );



	/* Creation of toggleButton_YSkipCustom */
	toggleButton_YSkipCustom = XtVaCreateManagedWidget( "toggleButton_YSkipCustom",
			xmToggleButtonWidgetClass,
			rowColumn_YSkip,
			RES_CONVERT( XmNlabelString, "Custom" ),
			NULL );
	XtAddCallback( toggleButton_YSkipCustom, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_YSkipCustom,
		(XtPointer) NULL );



	/* Creation of textField_YSkip */
	textField_YSkip = XtVaCreateManagedWidget( "textField_YSkip",
			xmTextFieldWidgetClass,
			form_VectorOptions,
			XmNsensitive, TRUE,
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 18,
			XmNleftWidget, rowColumn_YSkip,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 5,
			XmNtopWidget, textField_XSkip,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNwidth, 100,
			NULL );
	XtAddCallback( textField_YSkip, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_YSkip,
		(XtPointer) NULL );
	XtAddCallback( textField_YSkip, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_YSkip,
		(XtPointer) NULL );
	XtAddCallback( textField_YSkip, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField_YSkip,
		(XtPointer) NULL );



	/* Creation of frame_2DOptions */
	frame_2DOptions = XtVaCreateManagedWidget( "frame_2DOptions",
			xmFrameWidgetClass,
			form18,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNheight, 185,
			NULL );


	/* Creation of label_2DOptions */
	label_2DOptions = XtVaCreateManagedWidget( "label_2DOptions",
			xmLabelWidgetClass,
			frame_2DOptions,
			XmNx, 12,
			XmNy, 0,
			XmNwidth, 172,
			XmNheight, 14,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Shade/Contour/Filled Options" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_2DOptions */
	form_2DOptions = XtVaCreateManagedWidget( "form_2DOptions",
			xmFormWidgetClass,
			frame_2DOptions,
			XmNresizePolicy, XmRESIZE_GROW,
			NULL );


	/* Creation of rowColumn_Levels */
	rowColumn_Levels = XtVaCreateManagedWidget( "rowColumn_Levels",
			xmRowColumnWidgetClass,
			form_2DOptions,
			XmNnumColumns, 5,
			XmNorientation, XmVERTICAL,
			XmNradioBehavior, TRUE,
			XmNmarginHeight, 0,
			XmNradioAlwaysOne, TRUE,
			XmNwhichButton, 0,
			XmNpacking, XmPACK_TIGHT,
			XmNspacing, 1,
			XmNborderWidth, 0,
			XmNsensitive, TRUE,
			XmNadjustLast, TRUE,
			XmNadjustMargin, TRUE,
			XmNentryBorder, 0,
			XmNmarginWidth, 0,
			XmNshadowThickness, 0,
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 0,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of toggleButton_LevelsAutoscale */
	toggleButton_LevelsAutoscale = XtVaCreateManagedWidget( "toggleButton_LevelsAutoscale",
			xmToggleButtonWidgetClass,
			rowColumn_Levels,
			RES_CONVERT( XmNlabelString, "Autoscale Levels" ),
			XmNmarginBottom, 0,
			XmNmarginHeight, 0,
			XmNmarginTop, 0,
			XmNmarginWidth, 0,
			XmNspacing, 5,
			NULL );
	XtAddCallback( toggleButton_LevelsAutoscale, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_LevelsAutoscale,
		(XtPointer) NULL );



	/* Creation of toggleButton_LevelsReuse */
	toggleButton_LevelsReuse = XtVaCreateManagedWidget( "toggleButton_LevelsReuse",
			xmToggleButtonWidgetClass,
			rowColumn_Levels,
			RES_CONVERT( XmNlabelString, "Reuse Last Levels" ),
			XmNmarginBottom, 0,
			XmNmarginHeight, 0,
			XmNmarginTop, 0,
			XmNmarginWidth, 0,
			XmNspacing, 5,
			NULL );
	XtAddCallback( toggleButton_LevelsReuse, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_LevelsReuse,
		(XtPointer) NULL );



	/* Creation of toggleButton_LevelsCustom */
	toggleButton_LevelsCustom = XtVaCreateManagedWidget( "toggleButton_LevelsCustom",
			xmToggleButtonWidgetClass,
			rowColumn_Levels,
			RES_CONVERT( XmNlabelString, "Custom Levels" ),
			XmNmarginBottom, 0,
			XmNmarginHeight, 0,
			XmNmarginTop, 0,
			XmNmarginWidth, 0,
			XmNspacing, 5,
			NULL );
	XtAddCallback( toggleButton_LevelsCustom, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_LevelsCustom,
		(XtPointer) NULL );



	/* Creation of label_SCFLow */
	label_SCFLow = XtVaCreateManagedWidget( "label_SCFLow",
			xmLabelWidgetClass,
			form_2DOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Low" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 12,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_Levels,
			NULL );


	/* Creation of textField_SCFLow */
	textField_SCFLow = XtVaCreateManagedWidget( "textField_SCFLow",
			xmTextFieldWidgetClass,
			form_2DOptions,
			XmNsensitive, TRUE,
			XmNvalue, "",
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 3,
			XmNleftWidget, label_SCFLow,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNwidth, 80,
			XmNy, 70,
			XmNheight, 32,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 5,
			XmNtopWidget, rowColumn_Levels,
			NULL );
	XtAddCallback( textField_SCFLow, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_SCFLow,
		(XtPointer) NULL );
	XtAddCallback( textField_SCFLow, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_SCFLow,
		(XtPointer) NULL );



	/* Creation of label_SCFHigh */
	label_SCFHigh = XtVaCreateManagedWidget( "label_SCFHigh",
			xmLabelWidgetClass,
			form_2DOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "High" ),
			XmNleftOffset, 3,
			XmNleftWidget, textField_SCFLow,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 12,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_Levels,
			NULL );


	/* Creation of textField_SCFHigh */
	textField_SCFHigh = XtVaCreateManagedWidget( "textField_SCFHigh",
			xmTextFieldWidgetClass,
			form_2DOptions,
			XmNsensitive, TRUE,
			XmNvalue, "",
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 3,
			XmNleftWidget, label_SCFHigh,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 5,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNwidth, 80,
			XmNtopWidget, rowColumn_Levels,
			NULL );
	XtAddCallback( textField_SCFHigh, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_SCFHigh,
		(XtPointer) NULL );
	XtAddCallback( textField_SCFHigh, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_SCFHigh,
		(XtPointer) NULL );



	/* Creation of label_SCFDelta */
	label_SCFDelta = XtVaCreateManagedWidget( "label_SCFDelta",
			xmLabelWidgetClass,
			form_2DOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Delta" ),
			XmNleftOffset, 3,
			XmNleftWidget, textField_SCFHigh,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 12,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_Levels,
			NULL );


	/* Creation of textField_SCFDelta */
	textField_SCFDelta = XtVaCreateManagedWidget( "textField_SCFDelta",
			xmTextFieldWidgetClass,
			form_2DOptions,
			XmNsensitive, TRUE,
			XmNvalue, "",
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 3,
			XmNleftWidget, label_SCFDelta,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 5,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNwidth, 80,
			XmNtopWidget, rowColumn_Levels,
			NULL );
	XtAddCallback( textField_SCFDelta, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField_SCFDelta,
		(XtPointer) NULL );
	XtAddCallback( textField_SCFDelta, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField_SCFDelta,
		(XtPointer) NULL );



	/* Creation of toggleButton_SCFColorKey */
	toggleButton_SCFColorKey = XtVaCreateManagedWidget( "toggleButton_SCFColorKey",
			xmToggleButtonWidgetClass,
			form_2DOptions,
			RES_CONVERT( XmNlabelString, "Color Key" ),
			XmNalignment, XmALIGNMENT_BEGINNING,
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 5,
			XmNtopWidget, textField_SCFLow,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNwidth, 184,
			NULL );
	XtAddCallback( toggleButton_SCFColorKey, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_SCFColorKey,
		(XtPointer) NULL );



	/* Creation of toggleButton_SCFOverlayContours */
	toggleButton_SCFOverlayContours = XtVaCreateManagedWidget( "toggleButton_SCFOverlayContours",
			xmToggleButtonWidgetClass,
			form_2DOptions,
			RES_CONVERT( XmNlabelString, "Overlay Contours (Shade/Fill only)" ),
			XmNalignment, XmALIGNMENT_BEGINNING,
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 5,
			XmNtopWidget, toggleButton_SCFColorKey,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 5,
			NULL );
	XtAddCallback( toggleButton_SCFOverlayContours, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_SCFOverlayContours,
		(XtPointer) NULL );



	/* Creation of frame_1DOptions */
	frame_1DOptions = XtVaCreateManagedWidget( "frame_1DOptions",
			xmFrameWidgetClass,
			form18,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNheight, 145,
			NULL );


	/* Creation of label_1DOptions */
	label_1DOptions = XtVaCreateManagedWidget( "label_1DOptions",
			xmLabelWidgetClass,
			frame_1DOptions,
			RES_CONVERT( XmNlabelString, "Line/Scatter Options" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_1DOptions */
	form_1DOptions = XtVaCreateManagedWidget( "form_1DOptions",
			xmFormWidgetClass,
			frame_1DOptions,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 1,
			XmNy, 19,
			NULL );


	/* Creation of toggleButton_LSAutoSymbols */
	toggleButton_LSAutoSymbols = XtVaCreateManagedWidget( "toggleButton_LSAutoSymbols",
			xmToggleButtonWidgetClass,
			form_1DOptions,
			RES_CONVERT( XmNlabelString, "Automatic Lines/Symbols" ),
			XmNalignment, XmALIGNMENT_BEGINNING,
			XmNsensitive, TRUE,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			NULL );
	XtAddCallback( toggleButton_LSAutoSymbols, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_LSAutoSymbols,
		(XtPointer) NULL );



	/* Creation of label_LSLineStyle */
	label_LSLineStyle = XtVaCreateManagedWidget( "label_LSLineStyle",
			xmLabelWidgetClass,
			form_1DOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Line Style:" ),
			XmNleftOffset, 25,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 50,
			XmNtopAttachment, XmATTACH_FORM,
			XmNx, 25,
			NULL );


	/* Creation of label_LSSymbol */
	label_LSSymbol = XtVaCreateManagedWidget( "label_LSSymbol",
			xmLabelWidgetClass,
			form_1DOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Symbol:" ),
			XmNleftOffset, 25,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 25,
			XmNtopWidget, label_LSLineStyle,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of lineStyle2 */
	lineStyle2_shell = XtVaCreatePopupShell ("lineStyle2_shell",
			xmMenuShellWidgetClass, form_1DOptions,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	lineStyle2 = XtVaCreateWidget( "lineStyle2",
			xmRowColumnWidgetClass,
			lineStyle2_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 135,
			XmNy, 0,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNlabelString, "n.a." ),
			NULL );


	/* Creation of lineStyle2_b8 */
	lineStyle2_b8 = XtVaCreateManagedWidget( "lineStyle2_b8",
			xmPushButtonWidgetClass,
			lineStyle2,
			RES_CONVERT( XmNlabelString, "Auto/None" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle2_b8, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle2_b8,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b9 */
	lineStyle1_b9 = XtVaCreateManagedWidget( "lineStyle1_b9",
			xmPushButtonWidgetClass,
			lineStyle2,
			RES_CONVERT( XmNlabelString, "1" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b9, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b9,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b10 */
	lineStyle1_b10 = XtVaCreateManagedWidget( "lineStyle1_b10",
			xmPushButtonWidgetClass,
			lineStyle2,
			RES_CONVERT( XmNlabelString, "2" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b10, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b10,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b11 */
	lineStyle1_b11 = XtVaCreateManagedWidget( "lineStyle1_b11",
			xmPushButtonWidgetClass,
			lineStyle2,
			RES_CONVERT( XmNlabelString, "3" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b11, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b11,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b12 */
	lineStyle1_b12 = XtVaCreateManagedWidget( "lineStyle1_b12",
			xmPushButtonWidgetClass,
			lineStyle2,
			RES_CONVERT( XmNlabelString, "4" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b12, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b12,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b13 */
	lineStyle1_b13 = XtVaCreateManagedWidget( "lineStyle1_b13",
			xmPushButtonWidgetClass,
			lineStyle2,
			RES_CONVERT( XmNlabelString, "5" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b13, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b13,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b14 */
	lineStyle1_b14 = XtVaCreateManagedWidget( "lineStyle1_b14",
			xmPushButtonWidgetClass,
			lineStyle2,
			RES_CONVERT( XmNlabelString, "6" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b14, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b14,
		(XtPointer) NULL );



	/* Creation of optionMenu_LSLineStyle */
	optionMenu_LSLineStyle = XtVaCreateManagedWidget( "optionMenu_LSLineStyle",
			xmRowColumnWidgetClass,
			form_1DOptions,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, lineStyle2,
			XmNx, 121,
			XmNy, 35,
			XmNwidth, 70,
			XmNheight, 18,
			XmNsensitive, TRUE,
			XmNnumColumns, 3,
			XmNtearOffModel, XmTEAR_OFF_ENABLED,
			XmNpacking, XmPACK_COLUMN,
			XmNwhichButton, 5,
			XmNtopOffset, 0,
			XmNtopWidget, toggleButton_LSAutoSymbols,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftOffset, 0,
			XmNleftWidget, label_LSLineStyle,
			XmNleftAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of lineStyle3 */
	lineStyle3_shell = XtVaCreatePopupShell ("lineStyle3_shell",
			xmMenuShellWidgetClass, form_1DOptions,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	lineStyle3 = XtVaCreateWidget( "lineStyle3",
			xmRowColumnWidgetClass,
			lineStyle3_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 135,
			XmNy, 0,
			XmNsensitive, TRUE,
			XmNnumColumns, 4,
			XmNorientation, XmHORIZONTAL,
			XmNpacking, XmPACK_COLUMN,
			RES_CONVERT( XmNlabelString, "n.a." ),
			NULL );


	/* Creation of lineStyle_b1 */
	lineStyle_b1 = XtVaCreateManagedWidget( "lineStyle_b1",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "Auto" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b1,
		(XtPointer) NULL );



	/* Creation of optionMenu_p_b10 */
	optionMenu_p_b10 = XtVaCreateManagedWidget( "optionMenu_p_b10",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "1" ),
			XmNx, 139,
			XmNy, 2,
			XmNsensitive, TRUE,
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( optionMenu_p_b10, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p_b10,
		(XtPointer) NULL );



	/* Creation of lineStyle_b89 */
	lineStyle_b89 = XtVaCreateManagedWidget( "lineStyle_b89",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "2" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b89, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b89,
		(XtPointer) NULL );



	/* Creation of lineStyle_b91 */
	lineStyle_b91 = XtVaCreateManagedWidget( "lineStyle_b91",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "3" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b91, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b91,
		(XtPointer) NULL );



	/* Creation of lineStyle_b92 */
	lineStyle_b92 = XtVaCreateManagedWidget( "lineStyle_b92",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "4" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b92, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b92,
		(XtPointer) NULL );



	/* Creation of lineStyle_b93 */
	lineStyle_b93 = XtVaCreateManagedWidget( "lineStyle_b93",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "5" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b93, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b93,
		(XtPointer) NULL );



	/* Creation of lineStyle_b94 */
	lineStyle_b94 = XtVaCreateManagedWidget( "lineStyle_b94",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "6" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b94, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b94,
		(XtPointer) NULL );



	/* Creation of lineStyle_b95 */
	lineStyle_b95 = XtVaCreateManagedWidget( "lineStyle_b95",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "7" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b95, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b95,
		(XtPointer) NULL );



	/* Creation of lineStyle_b96 */
	lineStyle_b96 = XtVaCreateManagedWidget( "lineStyle_b96",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "8" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b96, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b96,
		(XtPointer) NULL );



	/* Creation of lineStyle_b97 */
	lineStyle_b97 = XtVaCreateManagedWidget( "lineStyle_b97",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "9" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b97, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b97,
		(XtPointer) NULL );



	/* Creation of lineStyle_b98 */
	lineStyle_b98 = XtVaCreateManagedWidget( "lineStyle_b98",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "10" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b98, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b98,
		(XtPointer) NULL );



	/* Creation of lineStyle_b99 */
	lineStyle_b99 = XtVaCreateManagedWidget( "lineStyle_b99",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "11" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b99, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b99,
		(XtPointer) NULL );



	/* Creation of lineStyle_b100 */
	lineStyle_b100 = XtVaCreateManagedWidget( "lineStyle_b100",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "12" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b100, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b100,
		(XtPointer) NULL );



	/* Creation of lineStyle_b101 */
	lineStyle_b101 = XtVaCreateManagedWidget( "lineStyle_b101",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "13" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b101, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b101,
		(XtPointer) NULL );



	/* Creation of lineStyle_b102 */
	lineStyle_b102 = XtVaCreateManagedWidget( "lineStyle_b102",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "14" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b102, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b102,
		(XtPointer) NULL );



	/* Creation of lineStyle_b103 */
	lineStyle_b103 = XtVaCreateManagedWidget( "lineStyle_b103",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "15" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b103, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b103,
		(XtPointer) NULL );



	/* Creation of lineStyle_b104 */
	lineStyle_b104 = XtVaCreateManagedWidget( "lineStyle_b104",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "16" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b104, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b104,
		(XtPointer) NULL );



	/* Creation of lineStyle_b105 */
	lineStyle_b105 = XtVaCreateManagedWidget( "lineStyle_b105",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "17" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b105, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b105,
		(XtPointer) NULL );



	/* Creation of lineStyle_b106 */
	lineStyle_b106 = XtVaCreateManagedWidget( "lineStyle_b106",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "18" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b106, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b106,
		(XtPointer) NULL );



	/* Creation of lineStyle_b107 */
	lineStyle_b107 = XtVaCreateManagedWidget( "lineStyle_b107",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "19" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b107, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b107,
		(XtPointer) NULL );



	/* Creation of lineStyle_b108 */
	lineStyle_b108 = XtVaCreateManagedWidget( "lineStyle_b108",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "20" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b108, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b108,
		(XtPointer) NULL );



	/* Creation of lineStyle_b109 */
	lineStyle_b109 = XtVaCreateManagedWidget( "lineStyle_b109",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "21" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b109, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b109,
		(XtPointer) NULL );



	/* Creation of lineStyle_b110 */
	lineStyle_b110 = XtVaCreateManagedWidget( "lineStyle_b110",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "22" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b110, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b110,
		(XtPointer) NULL );



	/* Creation of lineStyle_b111 */
	lineStyle_b111 = XtVaCreateManagedWidget( "lineStyle_b111",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "23" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b111, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b111,
		(XtPointer) NULL );



	/* Creation of lineStyle_b112 */
	lineStyle_b112 = XtVaCreateManagedWidget( "lineStyle_b112",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "24" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b112, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b112,
		(XtPointer) NULL );



	/* Creation of lineStyle_b113 */
	lineStyle_b113 = XtVaCreateManagedWidget( "lineStyle_b113",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "25" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b113, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b113,
		(XtPointer) NULL );



	/* Creation of lineStyle_b114 */
	lineStyle_b114 = XtVaCreateManagedWidget( "lineStyle_b114",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "26" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b114, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b114,
		(XtPointer) NULL );



	/* Creation of lineStyle_b115 */
	lineStyle_b115 = XtVaCreateManagedWidget( "lineStyle_b115",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "27" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b115, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b115,
		(XtPointer) NULL );



	/* Creation of lineStyle_b116 */
	lineStyle_b116 = XtVaCreateManagedWidget( "lineStyle_b116",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "28" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b116, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b116,
		(XtPointer) NULL );



	/* Creation of lineStyle_b117 */
	lineStyle_b117 = XtVaCreateManagedWidget( "lineStyle_b117",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "29" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b117, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b117,
		(XtPointer) NULL );



	/* Creation of lineStyle_b118 */
	lineStyle_b118 = XtVaCreateManagedWidget( "lineStyle_b118",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "30" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b118, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b118,
		(XtPointer) NULL );



	/* Creation of lineStyle_b119 */
	lineStyle_b119 = XtVaCreateManagedWidget( "lineStyle_b119",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "31" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b119, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b119,
		(XtPointer) NULL );



	/* Creation of lineStyle_b120 */
	lineStyle_b120 = XtVaCreateManagedWidget( "lineStyle_b120",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "32" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b120, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b120,
		(XtPointer) NULL );



	/* Creation of lineStyle_b121 */
	lineStyle_b121 = XtVaCreateManagedWidget( "lineStyle_b121",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "33" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b121, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b121,
		(XtPointer) NULL );



	/* Creation of lineStyle_b122 */
	lineStyle_b122 = XtVaCreateManagedWidget( "lineStyle_b122",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "34" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b122, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b122,
		(XtPointer) NULL );



	/* Creation of lineStyle_b123 */
	lineStyle_b123 = XtVaCreateManagedWidget( "lineStyle_b123",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "35" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b123, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b123,
		(XtPointer) NULL );



	/* Creation of lineStyle_b124 */
	lineStyle_b124 = XtVaCreateManagedWidget( "lineStyle_b124",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "36" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b124, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b124,
		(XtPointer) NULL );



	/* Creation of lineStyle_b125 */
	lineStyle_b125 = XtVaCreateManagedWidget( "lineStyle_b125",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "37" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b125, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b125,
		(XtPointer) NULL );



	/* Creation of lineStyle_b126 */
	lineStyle_b126 = XtVaCreateManagedWidget( "lineStyle_b126",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "38" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b126, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b126,
		(XtPointer) NULL );



	/* Creation of lineStyle_b127 */
	lineStyle_b127 = XtVaCreateManagedWidget( "lineStyle_b127",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "39" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b127, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b127,
		(XtPointer) NULL );



	/* Creation of lineStyle_b128 */
	lineStyle_b128 = XtVaCreateManagedWidget( "lineStyle_b128",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "40" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b128, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b128,
		(XtPointer) NULL );



	/* Creation of lineStyle_b129 */
	lineStyle_b129 = XtVaCreateManagedWidget( "lineStyle_b129",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "41" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b129, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b129,
		(XtPointer) NULL );



	/* Creation of lineStyle_b130 */
	lineStyle_b130 = XtVaCreateManagedWidget( "lineStyle_b130",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "42" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b130, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b130,
		(XtPointer) NULL );



	/* Creation of lineStyle_b131 */
	lineStyle_b131 = XtVaCreateManagedWidget( "lineStyle_b131",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "43" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b131, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b131,
		(XtPointer) NULL );



	/* Creation of lineStyle_b132 */
	lineStyle_b132 = XtVaCreateManagedWidget( "lineStyle_b132",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "44" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b132, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b132,
		(XtPointer) NULL );



	/* Creation of lineStyle_b133 */
	lineStyle_b133 = XtVaCreateManagedWidget( "lineStyle_b133",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "45" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b133, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b133,
		(XtPointer) NULL );



	/* Creation of lineStyle_b134 */
	lineStyle_b134 = XtVaCreateManagedWidget( "lineStyle_b134",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "46" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b134, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b134,
		(XtPointer) NULL );



	/* Creation of lineStyle_b135 */
	lineStyle_b135 = XtVaCreateManagedWidget( "lineStyle_b135",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "47" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b135, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b135,
		(XtPointer) NULL );



	/* Creation of lineStyle_b136 */
	lineStyle_b136 = XtVaCreateManagedWidget( "lineStyle_b136",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "48" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b136, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b136,
		(XtPointer) NULL );



	/* Creation of lineStyle_b137 */
	lineStyle_b137 = XtVaCreateManagedWidget( "lineStyle_b137",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "49" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b137, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b137,
		(XtPointer) NULL );



	/* Creation of lineStyle_b138 */
	lineStyle_b138 = XtVaCreateManagedWidget( "lineStyle_b138",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "50" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b138, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b138,
		(XtPointer) NULL );



	/* Creation of lineStyle_b139 */
	lineStyle_b139 = XtVaCreateManagedWidget( "lineStyle_b139",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "51" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b139, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b139,
		(XtPointer) NULL );



	/* Creation of lineStyle_b140 */
	lineStyle_b140 = XtVaCreateManagedWidget( "lineStyle_b140",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "52" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b140, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b140,
		(XtPointer) NULL );



	/* Creation of lineStyle_b141 */
	lineStyle_b141 = XtVaCreateManagedWidget( "lineStyle_b141",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "53" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b141, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b141,
		(XtPointer) NULL );



	/* Creation of lineStyle_b142 */
	lineStyle_b142 = XtVaCreateManagedWidget( "lineStyle_b142",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "54" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b142, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b142,
		(XtPointer) NULL );



	/* Creation of lineStyle_b143 */
	lineStyle_b143 = XtVaCreateManagedWidget( "lineStyle_b143",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "55" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b143, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b143,
		(XtPointer) NULL );



	/* Creation of lineStyle_b144 */
	lineStyle_b144 = XtVaCreateManagedWidget( "lineStyle_b144",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "56" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b144, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b144,
		(XtPointer) NULL );



	/* Creation of lineStyle_b145 */
	lineStyle_b145 = XtVaCreateManagedWidget( "lineStyle_b145",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "57" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b145, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b145,
		(XtPointer) NULL );



	/* Creation of lineStyle_b146 */
	lineStyle_b146 = XtVaCreateManagedWidget( "lineStyle_b146",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "58" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b146, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b146,
		(XtPointer) NULL );



	/* Creation of lineStyle_b147 */
	lineStyle_b147 = XtVaCreateManagedWidget( "lineStyle_b147",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "59" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b147, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b147,
		(XtPointer) NULL );



	/* Creation of lineStyle_b148 */
	lineStyle_b148 = XtVaCreateManagedWidget( "lineStyle_b148",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "60" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b148, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b148,
		(XtPointer) NULL );



	/* Creation of lineStyle_b149 */
	lineStyle_b149 = XtVaCreateManagedWidget( "lineStyle_b149",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "61" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b149, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b149,
		(XtPointer) NULL );



	/* Creation of lineStyle_b150 */
	lineStyle_b150 = XtVaCreateManagedWidget( "lineStyle_b150",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "62" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b150, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b150,
		(XtPointer) NULL );



	/* Creation of lineStyle_b151 */
	lineStyle_b151 = XtVaCreateManagedWidget( "lineStyle_b151",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "63" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b151, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b151,
		(XtPointer) NULL );



	/* Creation of lineStyle_b152 */
	lineStyle_b152 = XtVaCreateManagedWidget( "lineStyle_b152",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "64" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b152, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b152,
		(XtPointer) NULL );



	/* Creation of lineStyle_b153 */
	lineStyle_b153 = XtVaCreateManagedWidget( "lineStyle_b153",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "65" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b153, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b153,
		(XtPointer) NULL );



	/* Creation of lineStyle_b154 */
	lineStyle_b154 = XtVaCreateManagedWidget( "lineStyle_b154",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "66" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b154, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b154,
		(XtPointer) NULL );



	/* Creation of lineStyle_b155 */
	lineStyle_b155 = XtVaCreateManagedWidget( "lineStyle_b155",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "67" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b155, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b155,
		(XtPointer) NULL );



	/* Creation of lineStyle_b156 */
	lineStyle_b156 = XtVaCreateManagedWidget( "lineStyle_b156",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "68" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b156, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b156,
		(XtPointer) NULL );



	/* Creation of lineStyle_b157 */
	lineStyle_b157 = XtVaCreateManagedWidget( "lineStyle_b157",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "69" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b157, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b157,
		(XtPointer) NULL );



	/* Creation of lineStyle_b158 */
	lineStyle_b158 = XtVaCreateManagedWidget( "lineStyle_b158",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "70" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b158, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b158,
		(XtPointer) NULL );



	/* Creation of lineStyle_b159 */
	lineStyle_b159 = XtVaCreateManagedWidget( "lineStyle_b159",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "71" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b159, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b159,
		(XtPointer) NULL );



	/* Creation of lineStyle_b160 */
	lineStyle_b160 = XtVaCreateManagedWidget( "lineStyle_b160",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "72" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b160, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b160,
		(XtPointer) NULL );



	/* Creation of lineStyle_b161 */
	lineStyle_b161 = XtVaCreateManagedWidget( "lineStyle_b161",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "73" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b161, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b161,
		(XtPointer) NULL );



	/* Creation of lineStyle_b162 */
	lineStyle_b162 = XtVaCreateManagedWidget( "lineStyle_b162",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "74" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b162, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b162,
		(XtPointer) NULL );



	/* Creation of lineStyle_b163 */
	lineStyle_b163 = XtVaCreateManagedWidget( "lineStyle_b163",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "75" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b163, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b163,
		(XtPointer) NULL );



	/* Creation of lineStyle_b164 */
	lineStyle_b164 = XtVaCreateManagedWidget( "lineStyle_b164",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "76" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b164, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b164,
		(XtPointer) NULL );



	/* Creation of lineStyle_b165 */
	lineStyle_b165 = XtVaCreateManagedWidget( "lineStyle_b165",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "77" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b165, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b165,
		(XtPointer) NULL );



	/* Creation of lineStyle_b166 */
	lineStyle_b166 = XtVaCreateManagedWidget( "lineStyle_b166",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "78" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b166, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b166,
		(XtPointer) NULL );



	/* Creation of lineStyle_b167 */
	lineStyle_b167 = XtVaCreateManagedWidget( "lineStyle_b167",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "79" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b167, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b167,
		(XtPointer) NULL );



	/* Creation of lineStyle_b168 */
	lineStyle_b168 = XtVaCreateManagedWidget( "lineStyle_b168",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "80" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b168, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b168,
		(XtPointer) NULL );



	/* Creation of lineStyle_b169 */
	lineStyle_b169 = XtVaCreateManagedWidget( "lineStyle_b169",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "81" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b169, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b169,
		(XtPointer) NULL );



	/* Creation of lineStyle_b170 */
	lineStyle_b170 = XtVaCreateManagedWidget( "lineStyle_b170",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "82" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b170, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b170,
		(XtPointer) NULL );



	/* Creation of lineStyle_b171 */
	lineStyle_b171 = XtVaCreateManagedWidget( "lineStyle_b171",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "83" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b171, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b171,
		(XtPointer) NULL );



	/* Creation of lineStyle_b172 */
	lineStyle_b172 = XtVaCreateManagedWidget( "lineStyle_b172",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "84" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b172, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b172,
		(XtPointer) NULL );



	/* Creation of lineStyle_b173 */
	lineStyle_b173 = XtVaCreateManagedWidget( "lineStyle_b173",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "85" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b173, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b173,
		(XtPointer) NULL );



	/* Creation of lineStyle_b174 */
	lineStyle_b174 = XtVaCreateManagedWidget( "lineStyle_b174",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "86" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b174, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b174,
		(XtPointer) NULL );



	/* Creation of lineStyle_b175 */
	lineStyle_b175 = XtVaCreateManagedWidget( "lineStyle_b175",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "87" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b175, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b175,
		(XtPointer) NULL );



	/* Creation of lineStyle_b176 */
	lineStyle_b176 = XtVaCreateManagedWidget( "lineStyle_b176",
			xmPushButtonWidgetClass,
			lineStyle3,
			RES_CONVERT( XmNlabelString, "88" ),
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle_b176, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle_b176,
		(XtPointer) NULL );



	/* Creation of optionMenu_LSSymbol */
	optionMenu_LSSymbol = XtVaCreateManagedWidget( "optionMenu_LSSymbol",
			xmRowColumnWidgetClass,
			form_1DOptions,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, lineStyle3,
			XmNsensitive, TRUE,
			XmNspacing, 0,
			RES_CONVERT( XmNforeground, "black" ),
			RES_CONVERT( XmNlabelString, "" ),
			XmNtopOffset, 5,
			XmNtopWidget, optionMenu_LSLineStyle,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftOffset, 0,
			XmNleftWidget, optionMenu_LSLineStyle,
			XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET,
			NULL );


	/* Creation of label_LSThickness */
	label_LSThickness = XtVaCreateManagedWidget( "label_LSThickness",
			xmLabelWidgetClass,
			form_1DOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Thickness:" ),
			XmNleftOffset, 10,
			XmNleftWidget, optionMenu_LSLineStyle,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 50,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of lineStyle1 */
	lineStyle1_shell = XtVaCreatePopupShell ("lineStyle1_shell",
			xmMenuShellWidgetClass, form_1DOptions,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	lineStyle1 = XtVaCreateWidget( "lineStyle1",
			xmRowColumnWidgetClass,
			lineStyle1_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 135,
			XmNy, 0,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNlabelString, "n.a." ),
			NULL );


	/* Creation of lineStyle2_b1 */
	lineStyle2_b1 = XtVaCreateManagedWidget( "lineStyle2_b1",
			xmPushButtonWidgetClass,
			lineStyle1,
			RES_CONVERT( XmNlabelString, "0" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNlabelType, XmPIXMAP,
			NULL );
	XtAddCallback( lineStyle2_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle2_b1,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b1 */
	lineStyle1_b1 = XtVaCreateManagedWidget( "lineStyle1_b1",
			xmPushButtonWidgetClass,
			lineStyle1,
			RES_CONVERT( XmNlabelString, "7" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b1,
		(XtPointer) NULL );



	/* Creation of lineStyle1_b2 */
	lineStyle1_b2 = XtVaCreateManagedWidget( "lineStyle1_b2",
			xmPushButtonWidgetClass,
			lineStyle1,
			RES_CONVERT( XmNlabelString, "13" ),
			XmNlabelType, XmPIXMAP,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( lineStyle1_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle1_b2,
		(XtPointer) NULL );



	/* Creation of optionMenu_LSThickness */
	optionMenu_LSThickness = XtVaCreateManagedWidget( "optionMenu_LSThickness",
			xmRowColumnWidgetClass,
			form_1DOptions,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, lineStyle1,
			XmNsensitive, TRUE,
			XmNnumColumns, 3,
			XmNwhichButton, 5,
			XmNtopOffset, 0,
			XmNtopWidget, toggleButton_LSAutoSymbols,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftOffset, 0,
			XmNleftWidget, label_LSThickness,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNpacking, XmPACK_COLUMN,
			XmNtearOffModel, XmTEAR_OFF_ENABLED,
			NULL );


	/* Creation of lineStyle4 */
	lineStyle4_shell = XtVaCreatePopupShell ("lineStyle4_shell",
			xmMenuShellWidgetClass, form_1DOptions,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	lineStyle4 = XtVaCreateWidget( "lineStyle4",
			xmRowColumnWidgetClass,
			lineStyle4_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 135,
			XmNy, 0,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNlabelString, "n.a." ),
			NULL );


	/* Creation of lineStyle4_b2 */
	lineStyle4_b2 = XtVaCreateManagedWidget( "lineStyle4_b2",
			xmPushButtonWidgetClass,
			lineStyle4,
			RES_CONVERT( XmNlabelString, "n.a." ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of lineStyle2_b2 */
	lineStyle2_b2 = XtVaCreateManagedWidget( "lineStyle2_b2",
			xmPushButtonWidgetClass,
			lineStyle4,
			RES_CONVERT( XmNlabelString, "0" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNlabelType, XmPIXMAP,
			NULL );
	XtAddCallback( lineStyle2_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_lineStyle2_b2,
		(XtPointer) NULL );



	/* Creation of optionMenu18 */
	optionMenu18 = XtVaCreateManagedWidget( "optionMenu18",
			xmRowColumnWidgetClass,
			form_1DOptions,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, lineStyle4,
			XmNsensitive, FALSE,
			XmNnumColumns, 3,
			XmNtearOffModel, XmTEAR_OFF_ENABLED,
			XmNpacking, XmPACK_COLUMN,
			XmNorientation, XmHORIZONTAL,
			XmNwhichButton, 5,
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNleftWidget, label_LSThickness,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, toggleButton_LSAutoSymbols,
			NULL );


	/* Creation of pushButton_Dismiss */
	pushButton_Dismiss = XtVaCreateManagedWidget( "pushButton_Dismiss",
			xmPushButtonWidgetClass,
			form18,
			RES_CONVERT( XmNlabelString, "Dismiss" ),
			XmNleftOffset, -50,
			XmNleftPosition, 50,
			XmNheight, 30,
			XmNwidth, 100,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			NULL );
	XtAddCallback( pushButton_Dismiss, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton_Dismiss,
		(XtPointer) NULL );



	/* Creation of frame_SelectOptions */
	frame_SelectOptions = XtVaCreateManagedWidget( "frame_SelectOptions",
			xmFrameWidgetClass,
			form18,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomOffset, 10,
			XmNbottomWidget, pushButton_Dismiss,
			NULL );


	/* Creation of label_SelectOptions */
	label_SelectOptions = XtVaCreateManagedWidget( "label_SelectOptions",
			xmLabelWidgetClass,
			frame_SelectOptions,
			RES_CONVERT( XmNlabelString, "Select Options for:" ),
			XmNalignment, XmALIGNMENT_BEGINNING,
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_SelectOptions */
	form_SelectOptions = XtVaCreateManagedWidget( "form_SelectOptions",
			xmFormWidgetClass,
			frame_SelectOptions,
			XmNresizePolicy, XmRESIZE_GROW,
			NULL );


	/* Creation of rowColumn_PlotType */
	rowColumn_PlotType = XtVaCreateManagedWidget( "rowColumn_PlotType",
			xmRowColumnWidgetClass,
			form_SelectOptions,
			XmNnumColumns, 3,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNmarginHeight, 2,
			XmNradioAlwaysOne, TRUE,
			XmNwhichButton, 0,
			XmNmarginWidth, 2,
			XmNpacking, XmPACK_TIGHT,
			XmNisAligned, TRUE,
			XmNspacing, 2,
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 2,
			XmNtopAttachment, XmATTACH_FORM,
			XmNbottomOffset, 5,
			XmNbottomAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of toggleButton_PlotTypeLineScatter */
	toggleButton_PlotTypeLineScatter = XtVaCreateManagedWidget( "toggleButton_PlotTypeLineScatter",
			xmToggleButtonWidgetClass,
			rowColumn_PlotType,
			RES_CONVERT( XmNlabelString, "Line/Scatter" ),
			XmNunitType, XmPIXELS,
			NULL );
	XtAddCallback( toggleButton_PlotTypeLineScatter, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_PlotTypeLineScatter,
		(XtPointer) NULL );



	/* Creation of toggleButton_PlotTypeShadeContourFill */
	toggleButton_PlotTypeShadeContourFill = XtVaCreateManagedWidget( "toggleButton_PlotTypeShadeContourFill",
			xmToggleButtonWidgetClass,
			rowColumn_PlotType,
			RES_CONVERT( XmNlabelString, "Shade/Contour/Fill" ),
			NULL );
	XtAddCallback( toggleButton_PlotTypeShadeContourFill, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_PlotTypeShadeContourFill,
		(XtPointer) NULL );



	/* Creation of toggleButton_PlotTypeVector */
	toggleButton_PlotTypeVector = XtVaCreateManagedWidget( "toggleButton_PlotTypeVector",
			xmToggleButtonWidgetClass,
			rowColumn_PlotType,
			RES_CONVERT( XmNlabelString, "Vector" ),
			NULL );
	XtAddCallback( toggleButton_PlotTypeVector, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_PlotTypeVector,
		(XtPointer) NULL );



	/* Creation of frame_GeneralOptions */
	frame_GeneralOptions = XtVaCreateManagedWidget( "frame_GeneralOptions",
			xmFrameWidgetClass,
			form18,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNheight, 80,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomOffset, 5,
			XmNbottomWidget, frame_SelectOptions,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 5,
			XmNtopWidget, frame_2DOptions,
			NULL );


	/* Creation of label_GeneralOptions */
	label_GeneralOptions = XtVaCreateManagedWidget( "label_GeneralOptions",
			xmLabelWidgetClass,
			frame_GeneralOptions,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "General Options" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			NULL );


	/* Creation of form_GeneralOptions */
	form_GeneralOptions = XtVaCreateManagedWidget( "form_GeneralOptions",
			xmFormWidgetClass,
			frame_GeneralOptions,
			XmNresizePolicy, XmRESIZE_GROW,
			NULL );


	/* Creation of toggleButton_Transpose */
	toggleButton_Transpose = XtVaCreateManagedWidget( "toggleButton_Transpose",
			xmToggleButtonWidgetClass,
			form_GeneralOptions,
			RES_CONVERT( XmNlabelString, "Transpose" ),
			XmNalignment, XmALIGNMENT_BEGINNING,
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 2,
			XmNtopAttachment, XmATTACH_FORM,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 5,
			NULL );
	XtAddCallback( toggleButton_Transpose, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_Transpose,
		(XtPointer) NULL );



	/* Creation of toggleButton_NoLabels */
	toggleButton_NoLabels = XtVaCreateManagedWidget( "toggleButton_NoLabels",
			xmToggleButtonWidgetClass,
			form_GeneralOptions,
			RES_CONVERT( XmNlabelString, "No Labels" ),
			XmNtopOffset, 2,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNleftWidget, toggleButton_Transpose,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNalignment, XmALIGNMENT_BEGINNING,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 5,
			NULL );
	XtAddCallback( toggleButton_NoLabels, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton_NoLabels,
		(XtPointer) NULL );




	return ( PlotOptions );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_PlotOptions( swidget _UxUxParent )
{
	Widget                  rtrn;
	static int		_Uxinit = 0;

	UxParent = _UxUxParent;

	if ( ! _Uxinit )
	{
		XmRepTypeInstallTearOffModelConverter();
		UxLoadResources( "PlotOptions.rf" );
		_Uxinit = 1;
	}

	{
		if (gSavedPlotOptions== NULL) {
		rtrn = _Uxbuild_PlotOptions();

		InitArrays();
			InitPixmaps();
		}
		else
			rtrn = gSavedPlotOptions;
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		/* set the hi rez size */
		if (gHiRez && !gSavedPlotOptions) {
			Dimension width, height;
				
			XtVaGetValues(UxGetWidget(form18),
				XmNwidth, &width,
				XmNheight, &height,
				NULL);
			width = 1.2 * width;
			height = 1.1 * height;
				
			XtVaSetValues(UxGetWidget(form18),
				XmNwidth, width,
				XmNheight, height,
				NULL);
		}
		if (!gSavedPlotOptions)
			gSavedPlotOptions = rtrn;
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

