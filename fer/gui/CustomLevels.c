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
	CustomLevels.c

       Associated Header file: CustomLevels.h
*******************************************************************************/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
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

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include <Xm/FileSB.h>

#define CREATE	0
#define DELETE 1

/* prototypes */
static void InitialState(void);
static void ClearContourDisplay(void);
static void ClearOptions(void);
static void ClearWindow(void);
static void DisableAddRemove(void);
static void EnableAddRemove(void);
static void MaintainAddRemove(void);
static void GetDigit(char *digitText);
static void GetColor(char *colorText);
static void GetStyle(char *styleText);
extern void SaveOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void OpenOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
static void CreateContours(int mode);
extern void Cancel(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
static void SaveCTLFile(void);
static void OpenCTLFile(void);
swidget create_CustomLevels(swidget UxParent);
extern swidget create_Open_Save_ctl(swidget UxParent);

extern swidget Open_Save_ctl, fileSelectionBox2;
static int digitState = 0, styleState = 0;

/* globals */
swidget gSavedCustomLevels = NULL;
swidget CustomLevels;


static	Widget	form7;
static	Widget	label46;
static	Widget	bulletinBoard7;
static	Widget	label35;
static	Widget	label36;
static	Widget	label37;
static	Widget	rowColumn10;
static	Widget	toggleButton29;
static	Widget	toggleButton30;
static	Widget	toggleButton31;
static	Widget	optionMenu_p6_b40;
static	Widget	optionMenu_p6_b41;
static	Widget	optionMenu_p6_b42;
static	Widget	optionMenu_p6_b43;
static	Widget	optionMenu_p6_b44;
static	Widget	optionMenu21;
static	Widget	label41;
static	Widget	label42;
static	Widget	label43;
static	Widget	label44;
static	Widget	label45;
static	Widget	label40;
static	Widget	rowColumn11;
static	Widget	toggleButton33;
static	Widget	toggleButton34;
static	Widget	toggleButton35;
static	Widget	label47;
static	Widget	pushButton13;
static	Widget	pushButton14;
static	Widget	pushButton15;
static	Widget	pushButton16;
static	Widget	pushButton17;
static	Widget	label38;
static	Widget	scrolledWindowText6;
static	Widget	scrolledText5;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "CustomLevels.h"
#undef CONTEXT_MACRO_ACCESS

Widget	CustomLevels;
Widget	frame7;
Widget	textField34;
Widget	textField36;
Widget	textField37;
Widget	optionMenu_p13;
Widget	optionMenu_p_b19;
Widget	textField35;
Widget	textField38;
Widget	textField39;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static int numOptions=0;
static float lows[100], highs[100], deltas[100];

static void InitialState()
{
	ClearWindow();
	MaintainAddRemove();
	ClearOptions();
}

static void ClearContourDisplay()
{
	/*char *nullStr

	nullStr = '\0';*/
	/* clear the user contour field */
	XtVaSetValues(UxGetWidget(scrolledText5),
		XmNvalue, "",
		NULL);
	
	/* set save button to be insensitive */
	XtSetSensitive(UxGetWidget(pushButton13), False);
	DisableAddRemove();
}

static void DisableAddRemove()
{
	XtSetSensitive(UxGetWidget(pushButton15), False);
	XtSetSensitive(UxGetWidget(pushButton17), False);
}

static void EnableAddRemove()
{
	XtSetSensitive(UxGetWidget(pushButton15), True);
	XtSetSensitive(UxGetWidget(pushButton17), True);
}

static void MaintainAddRemove()
{
	char *testStr = (char *)XtMalloc(32);

	strcpy(testStr, "");

	/* check whether the low, high, delta are filled in and if so, enable ADD/REMOVE */
	testStr = XmTextFieldGetString(UxGetWidget(textField34));
	if (strlen(testStr) == 0) {
		DisableAddRemove();
		XtFree(testStr);
		return;
	}

	testStr = XmTextFieldGetString(UxGetWidget(textField36));
	if (strlen(testStr) == 0) {
		DisableAddRemove();
		XtFree(testStr);
		return;
	}

	testStr = XmTextFieldGetString(UxGetWidget(textField37));
	if (strlen(testStr) == 0) {
		DisableAddRemove();
		XtFree(testStr);
		return;
	}
	EnableAddRemove();
	XtFree(testStr);
}

static void ClearOptions()
{
	char nullStr[2];
	
	nullStr[0] = '\0';

	/* clear the levels flds */
	XtVaSetValues(UxGetWidget(textField34),
		XmNvalue, nullStr,
		NULL);
	XtVaSetValues(UxGetWidget(textField36),
		XmNvalue, nullStr,
		NULL);
	XtVaSetValues(UxGetWidget(textField37),
		XmNvalue, nullStr,
		NULL);

	/* clear the prefix and suffix flds */
	XtVaSetValues(UxGetWidget(textField38),
		XmNvalue, nullStr,
		NULL);
	XtVaSetValues(UxGetWidget(textField39),
		XmNvalue, nullStr,
		NULL);

	/* clear and unmap the decimal fld */
	XtVaSetValues(UxGetWidget(textField35),
		XmNvalue, nullStr,
		NULL);
	XtUnmapWidget(UxGetWidget(textField35));

	/* reset the digit toggles */
	XtVaSetValues(UxGetWidget(toggleButton33),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton34),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton35),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(rowColumn11),
/*50*/		XmNmenuHistory, "",
		NULL);
	digitState = 0;

	/* reset the style toggles */
	XtVaSetValues(UxGetWidget(toggleButton29),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton30),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton31),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(rowColumn10),
		XmNmenuHistory, "",
		NULL);
	styleState = 0;

	/* reset color menu to black */
}

static void ClearWindow()
{
	ClearContourDisplay();
	ClearOptions();
}

/* read the digits and get decimals if needed */
static void GetDigit(digitText)
char *digitText;
{
	Widget activeButton = NULL;
	XmString buttonLabel = XmStringCreate("Button", XmSTRING_DEFAULT_CHARSET), 
		noneLabel = XmStringCreate("None", XmSTRING_DEFAULT_CHARSET),
		intLabel = XmStringCreate("Integer", XmSTRING_DEFAULT_CHARSET),
		decLabel = XmStringCreate("Decimals:", XmSTRING_DEFAULT_CHARSET);
	char decText[3];

	strcpy(decText, "");
	strcpy(digitText, "");
	if (digitState) {
		/* get the toggle button */
		XtVaGetValues(UxGetWidget(rowColumn11),
			XmNmenuHistory, &activeButton,
			NULL);
		XtVaGetValues(activeButton,
			XmNlabelString, &buttonLabel,
			NULL);
		if (XmStringCompare(buttonLabel, noneLabel))
			strcpy(digitText, "None");
		else if (XmStringCompare(buttonLabel, intLabel))
			strcpy(digitText, "Integer");
		else if (XmStringCompare(buttonLabel, decLabel)) {
			/* get the decimal field */
			XtVaGetValues(UxGetWidget(textField34),
				XmNvalue, decText,
				NULL);
			strcpy(digitText, decText);
			strcat(digitText, " Places");
		}
	}

	XmStringFree(buttonLabel);
	XmStringFree(noneLabel);
	XmStringFree(intLabel);
	XmStringFree(decLabel);
}

/* read the color */
static void GetColor(colorText)
char *colorText;
{
	strcpy(colorText, "Black");
}

/* read the style */
static void GetStyle(styleText)
char *styleText;
{
	Widget activeButton = NULL;
	XmString buttonLabel = XmStringCreate("Button", XmSTRING_DEFAULT_CHARSET), 
		dashLabel = XmStringCreate("Dash", XmSTRING_DEFAULT_CHARSET),
		darkLabel = XmStringCreate("Dark", XmSTRING_DEFAULT_CHARSET),
		solidLabel = XmStringCreate("Solid", XmSTRING_DEFAULT_CHARSET); 

	strcpy(styleText, "");
	if (styleState) {
		XtVaGetValues(UxGetWidget(rowColumn10),
			XmNmenuHistory, &activeButton,
			NULL);
		/* get the togglebutton title */
		XtVaGetValues(activeButton,
			XmNlabelString, &buttonLabel,
			NULL);
		if (XmStringCompare(buttonLabel, dashLabel))
			strcpy(styleText, "Dash");
		else if (XmStringCompare(buttonLabel, darkLabel))
			strcpy(styleText, "Dark");
		else if (XmStringCompare(buttonLabel, solidLabel))
			strcpy(styleText, "Solid");
	}

	XmStringFree(buttonLabel);
	XmStringFree(dashLabel);
	XmStringFree(darkLabel);
	XmStringFree(solidLabel);
}

static void CreateContours(mode)
int mode;
{
	char *currContents, *numStr, *toTextBuffer, digitText[15], colorText[10], 
		styleText[6], *preText, *sufText, tempText[512], decText[10];
	int numLines = 0;
	register int i;
	float low, high, delta, lineValues[100], newVal;

	toTextBuffer = (char *)XtMalloc(5000 * sizeof(char));
	currContents = (char *)XtMalloc(10000 * sizeof(char));
	preText = (char *)XtMalloc(16 * sizeof(char));
	sufText = (char *)XtMalloc(16 * sizeof(char));
	numStr = (char *)XtMalloc(32 * sizeof(char));

	strcpy(toTextBuffer, "");
	strcpy(currContents, "");
	strcpy(preText, "");
	strcpy(sufText, "");
	strcpy(numStr, "");
	strcpy(digitText, "");
	strcpy(colorText, "");
	strcpy(styleText, "");
	strcpy(decText, "");
	strcpy(tempText, "");

	/* get low, high, delta */
	numStr = XmTextFieldGetString(UxGetWidget(textField34));
	sscanf(numStr, "%f", &low);

	numStr = XmTextFieldGetString(UxGetWidget(textField36));
	sscanf(numStr, "%f", &high);

	numStr = XmTextFieldGetString(UxGetWidget(textField37));
	sscanf(numStr, "%f", &delta);

	/* should do some error testing here */

	/* create the line values */
	newVal = low;
	while (1) {
		newVal = low + (numLines * delta);
		if (newVal > high) break;
		lineValues[numLines++] = newVal;
	}

	/* read the digits and get decimals if needed */
	GetDigit(digitText);

	/* read the color */
	GetColor(colorText); 

	/* read the style */
	GetStyle(styleText);

	/* read the prefix */
	preText = XmTextFieldGetString(UxGetWidget(textField38));

	/* read the suffix */
	sufText = XmTextFieldGetString(UxGetWidget(textField39));

	/* now build a string for display in the contour display */
	for (i=0;i<numLines;i++) {
		strcpy(tempText, "");
		if (mode == CREATE)
			sprintf(tempText, "(+) %.3f %8s %8s %6s %10s %10s\n",  lineValues[i], 
				digitText, colorText, styleText, preText, sufText);
		else
			sprintf(tempText, "(-) %.3f %8s %8s %6s %10s %10s\n",  lineValues[i], 
				digitText, colorText, styleText, preText, sufText);
		strcat(toTextBuffer, tempText);
	}

	/* append this to the textfield */
	/* first get contents 
	currContents = XmTextGetString(scrolledText5);*/
	XtVaGetValues(UxGetWidget(scrolledText5),
		XmNvalue, &currContents,
		NULL);

	strcat(currContents, toTextBuffer);
	XtVaSetValues(UxGetWidget(scrolledText5),
		XmNvalue, currContents,
		NULL);

	ClearOptions();

	/* set save to sensitive */
	XtSetSensitive(UxGetWidget(pushButton13), True);

	XtFree(numStr);
	XtFree(preText);
	XtFree(sufText);
	XtFree(toTextBuffer);
	XtFree(currContents);
}

/* ok and cancel callbacks for fileSelectionBox2 */

extern void SaveOK(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	XmSelectionBoxCallbackStruct *cbInfo = (XmSelectionBoxCallbackStruct *)UxCallbackArg;
	char *pathName, *contents;
	FILE *outFile;
	int io;

	pathName = (char *)XtMalloc(cbInfo->length);
	strcpy(pathName, "");
	XmStringGetLtoR(cbInfo->value, XmSTRING_DEFAULT_CHARSET, &pathName);

	/* open the file */
	outFile = fopen(pathName, "w");
	
	/* get a pointer to contour text */
	contents = (char *)XtMalloc(5000);
	strcpy(contents, "");
	XtVaGetValues(UxGetWidget(scrolledText5),
		XmNvalue, &contents,
		NULL);	

	/* write text to file */
	io = fwrite(contents, sizeof(char), strlen(contents), outFile);

	/* close file */
	io = fclose(outFile);

	/* pop down the interface */
	XtPopdown(UxGetWidget(Open_Save_ctl));

	XtFree(pathName);
	XtFree(contents);
}

extern void OpenOK(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	XmSelectionBoxCallbackStruct *cbInfo = (XmSelectionBoxCallbackStruct *)UxCallbackArg;
	char *pathName, *fileContents, *currContents;
	FILE *inFile;
	int io;

	pathName = (char *)XtMalloc(cbInfo->length);
	strcpy(pathName, "");
	XmStringGetLtoR(cbInfo->value, XmSTRING_DEFAULT_CHARSET, &pathName);

	/* open the file */
	inFile = fopen(pathName, "r");
	
	/* create a buffer to store contour text */
	fileContents = (char *)XtMalloc(5000);
	strcpy(fileContents, "");
/*300 */
	/* read text from file */
	io = fread(fileContents, sizeof(char), 5000, inFile);

	/* close file */
	io = fclose(inFile); 

	/* append this the contents of field */
	currContents = (char *)XtMalloc(10000);
	strcpy(currContents, "");
	XtVaGetValues(UxGetWidget(scrolledText5),
		XmNvalue, currContents,
		NULL);

	strcat(currContents, fileContents);

	XtVaSetValues(UxGetWidget(scrolledText5),
		XmNvalue, currContents,
		NULL);

	/* pop down the interface */
	XtPopdown(UxGetWidget(Open_Save_ctl));

	XtFree(pathName);
	XtFree(fileContents);
	XtFree(currContents);
}
extern void Cancel(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	/* pop down the interface */
	XtPopdown(UxGetWidget(Open_Save_ctl));
}

static void SaveCTLFile()
{
	XmString dirMask;

	/* see if the interface has been created */
	Open_Save_ctl = create_Open_Save_ctl(NO_PARENT);
	
	XtVaSetValues(UxGetWidget(Open_Save_ctl),
		XmNtitle, "Save Contour Levels",
		NULL); 

	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNokCallback,
		SaveOK,
		NULL);

	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNcancelCallback,
		Cancel,
		NULL);

	dirMask = XmStringCreateSimple("*.ctl");

	XtVaSetValues(UxGetWidget(fileSelectionBox2),
		XmNdirMask, dirMask,
		NULL);

	/* apply the mask */
	XmFileSelectionDoSearch((Widget)UxGetWidget(fileSelectionBox2), 
		dirMask);

	/* popup Open file */
	XtPopup(UxGetWidget(Open_Save_ctl), XtGrabNone);
}

static void OpenCTLFile()
{
	XmString dirMask;

	/* see if the interface has been created */
	Open_Save_ctl = create_Open_Save_ctl(NO_PARENT);

	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNokCallback,
		OpenOK,
		NULL);
	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNcancelCallback,
		Cancel,
		NULL);

	dirMask = XmStringCreateSimple("*.ctl");

	XtVaSetValues(UxGetWidget(fileSelectionBox2),
		XmNdirMask, dirMask,
		NULL);

	/* apply the mask */
	XmFileSelectionDoSearch((Widget)UxGetWidget(fileSelectionBox2), 
		dirMask);

	XtVaSetValues(UxGetWidget(Open_Save_ctl),
		XmNtitle, "Open Contour Levels File",
		NULL);

	/* popup Open file */
	XtPopup(UxGetWidget(Open_Save_ctl), XtGrabNone);

	XmStringFree(dirMask);
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	popupCB_CustomLevels(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	InitialState();
}

static	void	destroyCB_CustomLevels(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedCustomLevels = NULL;
}

static	void	destroyCB_form7(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedCustomLevels = NULL;
}

static	void	valueChangedCB_textField34(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	MaintainAddRemove();
}

static	void	valueChangedCB_textField36(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	MaintainAddRemove();
}

static	void	valueChangedCB_textField37(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	MaintainAddRemove();
}

static	void	valueChangedCB_toggleButton29(
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
		styleState = 1;
	
	}
}

static	void	valueChangedCB_toggleButton30(
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
		styleState = 2;
	
	}
}

static	void	valueChangedCB_toggleButton31(
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
		styleState = 3;
	
	}
}

static	void	activateCB_optionMenu_p_b19(
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

static	void	activateCB_optionMenu_p6_b41(
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

static	void	activateCB_optionMenu_p6_b42(
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

static	void	activateCB_optionMenu_p6_b43(
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

static	void	activateCB_optionMenu_p6_b44(
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

static	void	valueChangedCB_toggleButton33(
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
		digitState = 1;
	
	}
}

static	void	valueChangedCB_toggleButton34(
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
		digitState = 2;
	
	}
}

static	void	valueChangedCB_toggleButton35(
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
		XtMapWidget(UxGetWidget(textField35));
		digitState = 3;
	}
	else
		XtUnmapWidget(UxGetWidget(textField35));
	}
}

static	void	activateCB_pushButton13(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	SaveCTLFile();
}

static	void	activateCB_pushButton14(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	OpenCTLFile();
}

static	void	activateCB_pushButton15(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	CreateContours(CREATE);
	}
}

static	void	activateCB_pushButton16(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	ClearWindow();
}

static	void	activateCB_pushButton17(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	CreateContours(DELETE);
	}
}

static	void	valueChangedCB_scrolledText5(
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

static	void	modifyVerifyCB_scrolledText5(
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

static	void	motionVerifyCB_scrolledText5(
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

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_CustomLevels()
{
	Widget		_UxParent;
	Widget		optionMenu_p13_shell;


	/* Creation of CustomLevels */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	CustomLevels = XtVaCreatePopupShell( "CustomLevels",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 772,
			XmNheight, 263,
			XmNx, 0,
			XmNy, 0,
			XmNiconName, "Custom Levels",
			XmNtitle, "Custom Levels",
			RES_CONVERT( XmNbackground, "#7d87aa" ),
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( CustomLevels, XmNpopupCallback,
		(XtCallbackProc) popupCB_CustomLevels,
		(XtPointer) NULL );
	XtAddCallback( CustomLevels, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_CustomLevels,
		(XtPointer) NULL );



	/* Creation of form7 */
	form7 = XtVaCreateManagedWidget( "form7",
			xmFormWidgetClass,
			CustomLevels,
			XmNwidth, 772,
			XmNheight, 263,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, -2,
			XmNy, 0,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( form7, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_form7,
		(XtPointer) NULL );



	/* Creation of frame7 */
	frame7 = XtVaCreateManagedWidget( "frame7",
			xmFrameWidgetClass,
			form7,
			XmNwidth, 425,
			XmNheight, 180,
			XmNx, 10,
			XmNy, 31,
			XmNmappedWhenManaged, TRUE,
			XmNbottomAttachment, XmATTACH_NONE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of label46 */
	label46 = XtVaCreateManagedWidget( "label46",
			xmLabelWidgetClass,
			frame7,
			XmNx, 16,
			XmNy, 0,
			XmNwidth, 135,
			XmNheight, 17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Contour Options" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of bulletinBoard7 */
	bulletinBoard7 = XtVaCreateManagedWidget( "bulletinBoard7",
			xmBulletinBoardWidgetClass,
			frame7,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNwidth, 430,
			XmNheight, 192,
			XmNx, 0,
			XmNy, 18,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of label35 */
	label35 = XtVaCreateManagedWidget( "label35",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 81,
			XmNy, 16,
			XmNwidth, 27,
			XmNheight, 17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Low:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of textField34 */
	textField34 = XtVaCreateManagedWidget( "textField34",
			xmTextFieldWidgetClass,
			bulletinBoard7,
			XmNwidth, 69,
			XmNx, 111,
			XmNy, 9,
			XmNheight, 30,
			XmNsensitive, TRUE,
			XmNvalue, "",
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( textField34, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField34,
		(XtPointer) NULL );



	/* Creation of textField36 */
	textField36 = XtVaCreateManagedWidget( "textField36",
			xmTextFieldWidgetClass,
			bulletinBoard7,
			XmNwidth, 69,
			XmNx, 225,
			XmNy, 9,
			XmNheight, 30,
			XmNsensitive, TRUE,
			XmNvalue, "",
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( textField36, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField36,
		(XtPointer) NULL );



	/* Creation of textField37 */
	textField37 = XtVaCreateManagedWidget( "textField37",
			xmTextFieldWidgetClass,
			bulletinBoard7,
			XmNwidth, 69,
			XmNx, 343,
			XmNy, 9,
			XmNheight, 30,
			XmNsensitive, TRUE,
			XmNvalue, "",
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( textField37, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField37,
		(XtPointer) NULL );



	/* Creation of label36 */
	label36 = XtVaCreateManagedWidget( "label36",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 189,
			XmNy, 16,
			XmNwidth, 36,
			XmNheight, 17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "High:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of label37 */
	label37 = XtVaCreateManagedWidget( "label37",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 298,
			XmNy, 16,
			XmNwidth, 42,
			XmNheight, 17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Delta:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of rowColumn10 */
	rowColumn10 = XtVaCreateManagedWidget( "rowColumn10",
			xmRowColumnWidgetClass,
			bulletinBoard7,
			XmNwidth, 84,
			XmNheight, 60,
			XmNx, 207,
			XmNy, 71,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNradioBehavior, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of toggleButton29 */
	toggleButton29 = XtVaCreateManagedWidget( "toggleButton29",
			xmToggleButtonWidgetClass,
			rowColumn10,
			XmNx, 0,
			XmNy, 0,
			XmNwidth, 62,
			XmNheight, 25,
			RES_CONVERT( XmNlabelString, "Dash" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( toggleButton29, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton29,
		(XtPointer) NULL );



	/* Creation of toggleButton30 */
	toggleButton30 = XtVaCreateManagedWidget( "toggleButton30",
			xmToggleButtonWidgetClass,
			rowColumn10,
			XmNx, 545,
			XmNy, 28,
			XmNwidth, 63,
			XmNheight, 13,
			RES_CONVERT( XmNlabelString, "Dark" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( toggleButton30, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton30,
		(XtPointer) NULL );



	/* Creation of toggleButton31 */
	toggleButton31 = XtVaCreateManagedWidget( "toggleButton31",
			xmToggleButtonWidgetClass,
			rowColumn10,
			XmNx, 545,
			XmNy, 56,
			XmNwidth, 84,
			XmNheight, 12,
			RES_CONVERT( XmNlabelString, "Solid" ),
			XmNmarginHeight, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( toggleButton31, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton31,
		(XtPointer) NULL );



	/* Creation of optionMenu_p13 */
	optionMenu_p13_shell = XtVaCreatePopupShell ("optionMenu_p13_shell",
			xmMenuShellWidgetClass, bulletinBoard7,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p13 = XtVaCreateWidget( "optionMenu_p13",
			xmRowColumnWidgetClass,
			optionMenu_p13_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			XmNx, 474,
			XmNy, 0,
			XmNheight, 26,
			XmNmappedWhenManaged, TRUE,
			XmNwidth, 155,
			RES_CONVERT( XmNbackground, "#7d87aa" ),
			NULL );


	/* Creation of optionMenu_p_b19 */
	optionMenu_p_b19 = XtVaCreateManagedWidget( "optionMenu_p_b19",
			xmPushButtonWidgetClass,
			optionMenu_p13,
			RES_CONVERT( XmNlabelString, "Black" ),
			XmNx, 476,
			XmNy, 2,
			XmNheight, 26,
			XmNmappedWhenManaged, TRUE,
			XmNwidth, 155,
			XmNlabelType, XmSTRING,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p_b19, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p_b19,
		(XtPointer) NULL );



	/* Creation of optionMenu_p6_b40 */
	optionMenu_p6_b40 = XtVaCreateManagedWidget( "optionMenu_p6_b40",
			xmPushButtonWidgetClass,
			optionMenu_p13,
			RES_CONVERT( XmNlabelString, "Red" ),
			XmNx, 476,
			XmNy, 23,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of optionMenu_p6_b41 */
	optionMenu_p6_b41 = XtVaCreateManagedWidget( "optionMenu_p6_b41",
			xmPushButtonWidgetClass,
			optionMenu_p13,
			RES_CONVERT( XmNlabelString, "Green" ),
			XmNx, 476,
			XmNy, 44,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p6_b41, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p6_b41,
		(XtPointer) NULL );



	/* Creation of optionMenu_p6_b42 */
	optionMenu_p6_b42 = XtVaCreateManagedWidget( "optionMenu_p6_b42",
			xmPushButtonWidgetClass,
			optionMenu_p13,
			RES_CONVERT( XmNlabelString, "Blue" ),
			XmNx, 476,
			XmNy, 65,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p6_b42, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p6_b42,
		(XtPointer) NULL );



	/* Creation of optionMenu_p6_b43 */
	optionMenu_p6_b43 = XtVaCreateManagedWidget( "optionMenu_p6_b43",
			xmPushButtonWidgetClass,
			optionMenu_p13,
			RES_CONVERT( XmNlabelString, "Cyan" ),
			XmNx, 476,
			XmNy, 86,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p6_b43, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p6_b43,
		(XtPointer) NULL );



	/* Creation of optionMenu_p6_b44 */
	optionMenu_p6_b44 = XtVaCreateManagedWidget( "optionMenu_p6_b44",
			xmPushButtonWidgetClass,
			optionMenu_p13,
			RES_CONVERT( XmNlabelString, "Magenta" ),
			XmNx, 476,
			XmNy, 107,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p6_b44, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p6_b44,
		(XtPointer) NULL );



	/* Creation of optionMenu21 */
	optionMenu21 = XtVaCreateManagedWidget( "optionMenu21",
			xmRowColumnWidgetClass,
			bulletinBoard7,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p13,
			XmNx, 96,
			XmNy, 70,
			XmNwidth, 155,
			XmNheight, 30,
			XmNmappedWhenManaged, TRUE,
			XmNtearOffModel, XmTEAR_OFF_ENABLED,
			RES_CONVERT( XmNlabelString, " " ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of textField35 */
	textField35 = XtVaCreateManagedWidget( "textField35",
			xmTextFieldWidgetClass,
			bulletinBoard7,
			XmNwidth, 36,
			XmNx, 92,
			XmNy, 124,
			XmNheight, 30,
			XmNsensitive, TRUE,
			XmNvalue, "",
			XmNmappedWhenManaged, FALSE,
			RES_CONVERT( XmNbackground, "#7d87aa" ),
			NULL );


	/* Creation of label41 */
	label41 = XtVaCreateManagedWidget( "label41",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 124,
			XmNy, 50,
			XmNwidth, 64,
			XmNheight, 18,
			RES_CONVERT( XmNlabelString, "Color" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of label42 */
	label42 = XtVaCreateManagedWidget( "label42",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 212,
			XmNy, 50,
			XmNwidth, 43,
			XmNheight, 18,
			RES_CONVERT( XmNlabelString, "Style" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of label43 */
	label43 = XtVaCreateManagedWidget( "label43",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 313,
			XmNy, 50,
			XmNwidth, 61,
			XmNheight, 18,
			RES_CONVERT( XmNlabelString, "Options" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of label44 */
	label44 = XtVaCreateManagedWidget( "label44",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 278,
			XmNy, 79,
			XmNwidth, 77,
			XmNheight, 17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Prefix Text:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of textField38 */
	textField38 = XtVaCreateManagedWidget( "textField38",
			xmTextFieldWidgetClass,
			bulletinBoard7,
			XmNwidth, 57,
			XmNx, 355,
			XmNy, 71,
			XmNheight, 30,
			XmNsensitive, TRUE,
			XmNvalue, "",
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of label45 */
	label45 = XtVaCreateManagedWidget( "label45",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 278,
			XmNy, 109,
			XmNwidth, 77,
			XmNheight, 17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Suffix Text:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of textField39 */
	textField39 = XtVaCreateManagedWidget( "textField39",
			xmTextFieldWidgetClass,
			bulletinBoard7,
			XmNwidth, 57,
			XmNx, 355,
			XmNy, 103,
			XmNheight, 30,
			XmNsensitive, TRUE,
			XmNvalue, "",
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of label40 */
	label40 = XtVaCreateManagedWidget( "label40",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 18,
			XmNy, 50,
			XmNwidth, 61,
			XmNheight, 18,
			RES_CONVERT( XmNlabelString, "Digits" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of rowColumn11 */
	rowColumn11 = XtVaCreateManagedWidget( "rowColumn11",
			xmRowColumnWidgetClass,
			bulletinBoard7,
			XmNwidth, 84,
			XmNheight, 60,
			XmNx, 10,
			XmNy, 71,
			XmNmarginHeight, 0,
			XmNmarginWidth, 0,
			XmNradioBehavior, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of toggleButton33 */
	toggleButton33 = XtVaCreateManagedWidget( "toggleButton33",
			xmToggleButtonWidgetClass,
			rowColumn11,
			XmNx, 354,
			XmNy, 0,
			XmNwidth, 69,
			XmNheight, 17,
			RES_CONVERT( XmNlabelString, "None" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( toggleButton33, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton33,
		(XtPointer) NULL );



	/* Creation of toggleButton34 */
	toggleButton34 = XtVaCreateManagedWidget( "toggleButton34",
			xmToggleButtonWidgetClass,
			rowColumn11,
			XmNx, 0,
			XmNy, 28,
			XmNwidth, 2,
			XmNheight, 25,
			RES_CONVERT( XmNlabelString, "Integer" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( toggleButton34, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton34,
		(XtPointer) NULL );



	/* Creation of toggleButton35 */
	toggleButton35 = XtVaCreateManagedWidget( "toggleButton35",
			xmToggleButtonWidgetClass,
			rowColumn11,
			XmNx, 354,
			XmNy, 56,
			XmNwidth, 84,
			XmNheight, 12,
			RES_CONVERT( XmNlabelString, "Decimals:" ),
			XmNmarginHeight, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( toggleButton35, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton35,
		(XtPointer) NULL );



	/* Creation of label47 */
	label47 = XtVaCreateManagedWidget( "label47",
			xmLabelWidgetClass,
			bulletinBoard7,
			XmNx, 20,
			XmNy, 15,
			XmNwidth, 61,
			XmNheight, 18,
			RES_CONVERT( XmNlabelString, "Levels:" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of pushButton13 */
	pushButton13 = XtVaCreateManagedWidget( "pushButton13",
			xmPushButtonWidgetClass,
			form7,
			XmNx, 377,
			XmNy, 221,
			XmNwidth, 90,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "Save..." ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			NULL );
	XtAddCallback( pushButton13, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton13,
		(XtPointer) NULL );



	/* Creation of pushButton14 */
	pushButton14 = XtVaCreateManagedWidget( "pushButton14",
			xmPushButtonWidgetClass,
			form7,
			XmNx, 264,
			XmNy, 222,
			XmNwidth, 90,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "Open..." ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			NULL );
	XtAddCallback( pushButton14, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton14,
		(XtPointer) NULL );



	/* Creation of pushButton15 */
	pushButton15 = XtVaCreateManagedWidget( "pushButton15",
			xmPushButtonWidgetClass,
			form7,
			XmNx, 453,
			XmNy, 77,
			XmNwidth, 90,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "Add >>" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( pushButton15, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton15,
		(XtPointer) NULL );



	/* Creation of pushButton16 */
	pushButton16 = XtVaCreateManagedWidget( "pushButton16",
			xmPushButtonWidgetClass,
			form7,
			XmNx, 453,
			XmNy, 141,
			XmNwidth, 90,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "<< Clear >>" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( pushButton16, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton16,
		(XtPointer) NULL );



	/* Creation of pushButton17 */
	pushButton17 = XtVaCreateManagedWidget( "pushButton17",
			xmPushButtonWidgetClass,
			form7,
			XmNx, 453,
			XmNy, 110,
			XmNwidth, 90,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "Remove >>" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( pushButton17, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton17,
		(XtPointer) NULL );



	/* Creation of label38 */
	label38 = XtVaCreateManagedWidget( "label38",
			xmLabelWidgetClass,
			form7,
			XmNx, 561,
			XmNy, 35,
			XmNwidth, 117,
			XmNheight, 17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Contour Values" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );


	/* Creation of scrolledWindowText6 */
	scrolledWindowText6 = XtVaCreateManagedWidget( "scrolledWindowText6",
			xmScrolledWindowWidgetClass,
			form7,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNx, 553,
			XmNy, 52,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNheight, 199,
			XmNwidth, 209,
			NULL );


	/* Creation of scrolledText5 */
	scrolledText5 = XtVaCreateManagedWidget( "scrolledText5",
			xmTextWidgetClass,
			scrolledWindowText6,
			XmNwidth, 185,
			XmNheight, 180,
			XmNeditMode, XmMULTI_LINE_EDIT ,
			RES_CONVERT( XmNbackground, "gray75" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNfontList, UxConvertFontList( "-adobe-courier-bold-r-normal--*-100-*-*-m-60-iso8859-1" ),
			NULL );
	XtAddCallback( scrolledText5, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrolledText5,
		(XtPointer) NULL );
	XtAddCallback( scrolledText5, XmNmodifyVerifyCallback,
		(XtCallbackProc) modifyVerifyCB_scrolledText5,
		(XtPointer) NULL );
	XtAddCallback( scrolledText5, XmNmotionVerifyCallback,
		(XtCallbackProc) motionVerifyCB_scrolledText5,
		(XtPointer) NULL );




	return ( CustomLevels );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_CustomLevels( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedCustomLevels == NULL) {
		rtrn = _Uxbuild_CustomLevels();

		gSavedCustomLevels = rtrn;
		}
		else
			rtrn = gSavedCustomLevels;
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

