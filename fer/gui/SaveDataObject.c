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
	SaveDataObject.c

       Associated Header file: SaveDataObject.h
*******************************************************************************/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
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

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "ferret_structures.h"

typedef struct {
        int autoOrCustomName;
        char customName[256];
        enum {text, binary, netcdf} fileFormat;
        int heading, fixedTimeSpan, customOrder, append, fortranFormat;
        char custOrder[5], ftnFormat[128];
} JC_CSO;
 
JC_CSO customSaveOptions;
 
/* globals */
swidget SaveDataObject, gSavedSaveDataObject = NULL;
extern Boolean gHiRez;
static char custOrderText[5]={""};
 
/* JC addition */
extern Widget toggleButton_Regridding;
extern JC_Variable GLOBAL_Variable;
extern JC_Regridding GLOBAL_Regridding;
extern JC_Region GLOBAL_Region;
extern void JC_ListFileCommand_Create( char *command, JC_Object *O_ptr, JC_CSO *CSO_ptr );
 
/* prototypes */
swidget create_SaveDataObject(swidget UxParent);
static void SetInitialState(void);
static void UnSetAll(void);
static void ToggleText(void);
static void ToggleUnf(void);
static void ToggleCdf(void);
static void DoCancel(void);
static void DoSave(void);
static void UpdateSaveOptions(void);
static void ShowSamples(void);
static void HideSamples(void);
static void ShowFormat(void);
static void HideFormat(void);
extern void CreateListCmnd(int mode);


static	Widget	form28;
static	Widget	rowColumn36;
static	Widget	toggleButton16;
static	Widget	toggleButton23;
static	Widget	rowColumn38;
static	Widget	pushButton36;
static	Widget	label39;
static	Widget	form29;
static	Widget	rowColumn35;
static	Widget	toggleButton13;
static	Widget	toggleButton14;
static	Widget	toggleButton15;
static	Widget	label51;
static	Widget	form30;
static	Widget	rowColumn37;
static	Widget	toggleButton24;
static	Widget	toggleButton25;
static	Widget	toggleButton26;
static	Widget	toggleButton27;
static	Widget	toggleButton28;
static	Widget	textField18;
static	Widget	optionMenu_p9;
static	Widget	optionMenu_p9_b2;
static	Widget	optionMenu_p9_b3;
static	Widget	optionMenu_p9_b4;
static	Widget	optionMenu_p9_b5;
static	Widget	optionMenu_p9_b6;
static	Widget	optionMenu_p9_b7;
static	Widget	optionMenu_p9_b8;
static	Widget	optionMenu_p9_b9;
static	Widget	optionMenu_p9_b10;
static	Widget	optionMenu_p9_b11;
static	Widget	optionMenu_p9_b12;
static	Widget	optionMenu_p9_b13;
static	Widget	optionMenu_p9_b14;
static	Widget	optionMenu_p9_b15;
static	Widget	optionMenu_p9_b16;
static	Widget	optionMenu_p9_b17;
static	Widget	optionMenu_p9_b18;
static	Widget	optionMenu_p9_b19;
static	Widget	optionMenu_p9_b20;
static	Widget	optionMenu_p9_b21;
static	Widget	optionMenu_p9_b22;
static	Widget	optionMenu_p9_b23;
static	Widget	optionMenu_p9_b24;
static	Widget	optionMenu_p9_b26;
static	Widget	optionMenu17;
static	Widget	textField1;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "SaveDataObject.h"
#undef CONTEXT_MACRO_ACCESS

Widget	SaveDataObject;
Widget	pushButton38;
Widget	frame6;
Widget	frame10;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static void UpdateSaveOptions()
{

}

static void HideSamples()
{
	XtUnmanageChild(optionMenu17);
}

static void ShowSamples()
{
	XtManageChild(optionMenu17);
}

static void HideFormat()
{
	XtVaSetValues(UxGetWidget(textField18),
		XmNvalue, "",
		NULL);
	XtUnmanageChild(textField18);
}

static void ShowFormat()
{
	XtVaSetValues(UxGetWidget(textField18),
		XmNvalue, "",
		NULL);
	XtManageChild(textField18);
}

static void InitialState()
{
	XmToggleButtonSetState(toggleButton16, True, False);
	XmToggleButtonSetState(toggleButton23, False, False);
	XtUnmanageChild(textField1);
	XmToggleButtonSetState(toggleButton13, True, False);
	XmToggleButtonSetState(toggleButton14, False, False);
	XmToggleButtonSetState(toggleButton15, False, False);
	ToggleText();
}

static void UnSetAll()
{
	XmToggleButtonSetState(toggleButton24, False, False);
	XmToggleButtonSetState(toggleButton25, False, False);
	XmToggleButtonSetState(toggleButton26, False, False);
	XmToggleButtonSetState(toggleButton27, False, False);
	XmToggleButtonSetState(toggleButton28, False, False);
}

static void ToggleText()
{
	HideFormat();
	HideSamples();
	UnSetAll();
	XtSetSensitive(toggleButton24, True);
	XtSetSensitive(toggleButton25, True);
	XtSetSensitive(toggleButton26, True);
	XtSetSensitive(toggleButton27, True);
	XtSetSensitive(toggleButton28, False);
}

static void ToggleUnf()
{
	HideFormat();
	HideSamples();
	UnSetAll();
	XtSetSensitive(toggleButton24, True);
	XtSetSensitive(toggleButton25, True);
	XtSetSensitive(toggleButton26, False);
	XtSetSensitive(toggleButton27, False);
	XtSetSensitive(toggleButton28, False);
}

static void ToggleCdf()
{
	HideFormat();
	HideSamples();
	UnSetAll();
	XtSetSensitive(toggleButton24, True);
	XtSetSensitive(toggleButton25, False);
	XtSetSensitive(toggleButton26, False);
	XtSetSensitive(toggleButton27, False);
	XtSetSensitive(toggleButton28, True);
}

static void AddSampleText(wid, client_data, cbs)
Widget wid;
XtPointer client_data;
XtPointer cbs;
{
	char *tempText;
	XmString buttonLabel;
	int val;
	
	/* sample is encoded in button label */
	XtVaGetValues(wid,
		XmNlabelString, &buttonLabel,
		NULL);
	XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
	strcpy(custOrderText, tempText);
	XtFree(tempText);
}

static void DoSave()
{
	char *ttext;
    char command[MAX_COMMAND_LENGTH];
    JC_Object Obj;

	if (XmToggleButtonGetState(toggleButton16))
		customSaveOptions.autoOrCustomName = True;
	else {
		customSaveOptions.autoOrCustomName = False;
		XtVaGetValues(textField1,
			XmNvalue, &ttext,
			NULL);
		if (strlen(ttext))
			strcpy(customSaveOptions.customName, ttext);
		else 
			customSaveOptions.autoOrCustomName = True;
	}

	if (XmToggleButtonGetState(toggleButton13))
		customSaveOptions.fileFormat = text;
	else if (XmToggleButtonGetState(toggleButton14))
		customSaveOptions.fileFormat = binary;
	else 
		customSaveOptions.fileFormat = netcdf;

	if (XmToggleButtonGetState(toggleButton24))
		customSaveOptions.append = True;
	else
		customSaveOptions.append = False;

	if (XmToggleButtonGetState(toggleButton25)) {
		customSaveOptions.customOrder = True;
		if (strlen(custOrderText))
			strcpy(customSaveOptions.custOrder, custOrderText);
		else
			customSaveOptions.customOrder = False;
	}
	else
		customSaveOptions.customOrder = False;

	if (XmToggleButtonGetState(toggleButton26))
		customSaveOptions.heading = True;
	else
		customSaveOptions.heading = False;

	if (XmToggleButtonGetState(toggleButton27)) {
		customSaveOptions.fortranFormat = True;
		XtVaGetValues(textField18,
			XmNvalue, &ttext,
			NULL);	
		if (strlen(ttext))
			strcpy(customSaveOptions.ftnFormat, ttext);
		else
			customSaveOptions.fortranFormat = False;
	}
	else
		customSaveOptions.fortranFormat = False;

	if (XmToggleButtonGetState(toggleButton28))
		customSaveOptions.fixedTimeSpan = True;
	else
		customSaveOptions.fixedTimeSpan = False;

     Obj.variable = GLOBAL_Variable;
     Obj.regridding = GLOBAL_Regridding;
     Obj.region = GLOBAL_Region;
     Obj.fixed_regridding = XmToggleButtonGetState(UxGetWidget(toggleButton_Regridding));
 
    JC_ListFileCommand_Create(command, &Obj, &customSaveOptions);
    ferret_command(command, IGNORE_COMMAND_WIDGET);
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	destroyCB_SaveDataObject(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedSaveDataObject = NULL;
}

static	void	popupCB_SaveDataObject(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	InitialState();
}

static	void	valueChangedCB_toggleButton16(
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
		XtUnmanageChild(textField1);
	}
	}
}

static	void	valueChangedCB_toggleButton23(
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
		XtManageChild(textField1);
	}
	}
}

static	void	activateCB_pushButton36(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XtPopdown(SaveDataObject);
	}
}

static	void	activateCB_pushButton38(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	DoSave();
	}
}

static	void	valueChangedCB_toggleButton13(
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
		ToggleText();
	}
	}
}

static	void	valueChangedCB_toggleButton14(
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
		ToggleUnf();
	}
	}
}

static	void	valueChangedCB_toggleButton15(
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
		ToggleCdf();
	}
	}
}

static	void	valueChangedCB_toggleButton24(
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
	;
	}
	UpdateSaveOptions();
	}
}

static	void	valueChangedCB_toggleButton25(
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
		ShowSamples();
	else 
		HideSamples();
	UpdateSaveOptions();
	}
}

static	void	valueChangedCB_toggleButton26(
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
	;
	}
	UpdateSaveOptions();
	}
}

static	void	valueChangedCB_toggleButton27(
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
		ShowFormat();
	else
		HideFormat();
	UpdateSaveOptions();
	}
}

static	void	valueChangedCB_toggleButton28(
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
	;
	}
	UpdateSaveOptions();
	}
}

static	void	activateCB_optionMenu_p9_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b3(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b4(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b5(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b6(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b7(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b8(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b10(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b11(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b12(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b13(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b14(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b15(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b16(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b17(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b18(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b19(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b20(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b21(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b22(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b23(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b24(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	activateCB_optionMenu_p9_b26(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	AddSampleText(UxWidget, UxClientData, UxCallbackArg);
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_SaveDataObject()
{
	Widget		_UxParent;
	Widget		optionMenu_p9_shell;


	/* Creation of SaveDataObject */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	SaveDataObject = XtVaCreatePopupShell( "SaveDataObject",
			topLevelShellWidgetClass,
			_UxParent,
			XmNx, 49,
			XmNy, 162,
			XmNwidth, 607,
			XmNheight, 282,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNiconName, "Ferret:  Save Data Object",
			XmNtitle, "Save Data Object",
			NULL );
	XtAddCallback( SaveDataObject, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_SaveDataObject,
		(XtPointer) NULL );
	XtAddCallback( SaveDataObject, XmNpopupCallback,
		(XtCallbackProc) popupCB_SaveDataObject,
		(XtPointer) NULL );



	/* Creation of form28 */
	form28 = XtVaCreateManagedWidget( "form28",
			xmFormWidgetClass,
			SaveDataObject,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNunitType, XmPIXELS,
			XmNx, 105,
			XmNy, 37,
			XmNwidth, 607,
			XmNheight, 282,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of rowColumn36 */
	rowColumn36 = XtVaCreateManagedWidget( "rowColumn36",
			xmRowColumnWidgetClass,
			form28,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNpacking, XmPACK_TIGHT,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNspacing, 5,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 15,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of toggleButton16 */
	toggleButton16 = XtVaCreateManagedWidget( "toggleButton16",
			xmToggleButtonWidgetClass,
			rowColumn36,
			RES_CONVERT( XmNlabelString, "Automatic Filenames" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton16, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton16,
		(XtPointer) NULL );



	/* Creation of toggleButton23 */
	toggleButton23 = XtVaCreateManagedWidget( "toggleButton23",
			xmToggleButtonWidgetClass,
			rowColumn36,
			RES_CONVERT( XmNlabelString, "Custom Filename:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton23, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton23,
		(XtPointer) NULL );



	/* Creation of rowColumn38 */
	rowColumn38 = XtVaCreateManagedWidget( "rowColumn38",
			xmRowColumnWidgetClass,
			form28,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNentryAlignment, XmALIGNMENT_CENTER,
			XmNorientation, XmHORIZONTAL,
			XmNpacking, XmPACK_COLUMN,
			XmNspacing, 3,
			XmNleftPosition, 39,
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomOffset, 10,
			XmNbottomAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of pushButton36 */
	pushButton36 = XtVaCreateManagedWidget( "pushButton36",
			xmPushButtonWidgetClass,
			rowColumn38,
			RES_CONVERT( XmNlabelString, "Cancel" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton36, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton36,
		(XtPointer) NULL );



	/* Creation of pushButton38 */
	pushButton38 = XtVaCreateManagedWidget( "pushButton38",
			xmPushButtonWidgetClass,
			rowColumn38,
			RES_CONVERT( XmNlabelString, "Save" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton38, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton38,
		(XtPointer) NULL );



	/* Creation of frame6 */
	frame6 = XtVaCreateManagedWidget( "frame6",
			xmFrameWidgetClass,
			form28,
			XmNmappedWhenManaged, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNx, 10,
			XmNy, 58,
			XmNwidth, 192,
			XmNheight, 164,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNtopWidget, rowColumn36,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNbottomOffset, 50,
			XmNbottomAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of label39 */
	label39 = XtVaCreateManagedWidget( "label39",
			xmLabelWidgetClass,
			frame6,
			RES_CONVERT( XmNlabelString, "File Format" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );


	/* Creation of form29 */
	form29 = XtVaCreateManagedWidget( "form29",
			xmFormWidgetClass,
			frame6,
			XmNresizePolicy, XmRESIZE_ANY,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNmarginHeight, 10,
			XmNverticalSpacing, 10,
			NULL );


	/* Creation of rowColumn35 */
	rowColumn35 = XtVaCreateManagedWidget( "rowColumn35",
			xmRowColumnWidgetClass,
			form29,
			XmNorientation, XmVERTICAL,
			XmNradioBehavior, TRUE,
			XmNpacking, XmPACK_TIGHT,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNspacing, 2,
			XmNtopOffset, 0,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of toggleButton13 */
	toggleButton13 = XtVaCreateManagedWidget( "toggleButton13",
			xmToggleButtonWidgetClass,
			rowColumn35,
			RES_CONVERT( XmNlabelString, "Text" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton13, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton13,
		(XtPointer) NULL );



	/* Creation of toggleButton14 */
	toggleButton14 = XtVaCreateManagedWidget( "toggleButton14",
			xmToggleButtonWidgetClass,
			rowColumn35,
			RES_CONVERT( XmNlabelString, "Unformatted" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton14, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton14,
		(XtPointer) NULL );



	/* Creation of toggleButton15 */
	toggleButton15 = XtVaCreateManagedWidget( "toggleButton15",
			xmToggleButtonWidgetClass,
			rowColumn35,
			RES_CONVERT( XmNlabelString, "netCDF" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton15, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton15,
		(XtPointer) NULL );



	/* Creation of frame10 */
	frame10 = XtVaCreateManagedWidget( "frame10",
			xmFrameWidgetClass,
			form28,
			XmNmappedWhenManaged, TRUE,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNx, 328,
			XmNy, 61,
			XmNwidth, 293,
			XmNheight, 154,
			XmNleftOffset, 10,
			XmNleftWidget, frame6,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 0,
			XmNtopWidget, frame6,
			XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNbottomOffset, 0,
			XmNbottomWidget, frame6,
			XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of label51 */
	label51 = XtVaCreateManagedWidget( "label51",
			xmLabelWidgetClass,
			frame10,
			RES_CONVERT( XmNlabelString, "Options" ),
			XmNchildType, XmFRAME_TITLE_CHILD,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );


	/* Creation of form30 */
	form30 = XtVaCreateManagedWidget( "form30",
			xmFormWidgetClass,
			frame10,
			XmNresizePolicy, XmRESIZE_ANY,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNmarginHeight, 10,
			XmNverticalSpacing, 10,
			XmNheight, 21,
			XmNy, 139,
			NULL );


	/* Creation of rowColumn37 */
	rowColumn37 = XtVaCreateManagedWidget( "rowColumn37",
			xmRowColumnWidgetClass,
			form30,
			XmNorientation, XmVERTICAL,
			XmNradioBehavior, FALSE,
			XmNpacking, XmPACK_TIGHT,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNspacing, 2,
			XmNtopOffset, 0,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of toggleButton24 */
	toggleButton24 = XtVaCreateManagedWidget( "toggleButton24",
			xmToggleButtonWidgetClass,
			rowColumn37,
			RES_CONVERT( XmNlabelString, "Append" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton24, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton24,
		(XtPointer) NULL );



	/* Creation of toggleButton25 */
	toggleButton25 = XtVaCreateManagedWidget( "toggleButton25",
			xmToggleButtonWidgetClass,
			rowColumn37,
			RES_CONVERT( XmNlabelString, "Custom Order:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton25, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton25,
		(XtPointer) NULL );



	/* Creation of toggleButton26 */
	toggleButton26 = XtVaCreateManagedWidget( "toggleButton26",
			xmToggleButtonWidgetClass,
			rowColumn37,
			RES_CONVERT( XmNlabelString, "Header" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton26, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton26,
		(XtPointer) NULL );



	/* Creation of toggleButton27 */
	toggleButton27 = XtVaCreateManagedWidget( "toggleButton27",
			xmToggleButtonWidgetClass,
			rowColumn37,
			RES_CONVERT( XmNlabelString, "FORTRAN Format:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton27, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton27,
		(XtPointer) NULL );



	/* Creation of toggleButton28 */
	toggleButton28 = XtVaCreateManagedWidget( "toggleButton28",
			xmToggleButtonWidgetClass,
			rowColumn37,
			RES_CONVERT( XmNlabelString, "Fixed Time Span" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton28, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton28,
		(XtPointer) NULL );



	/* Creation of textField18 */
	textField18 = XtVaCreateManagedWidget( "textField18",
			xmTextFieldWidgetClass,
			form30,
			XmNwidth, 82,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 0,
			XmNleftWidget, rowColumn37,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNbottomOffset, 29,
			XmNbottomWidget, rowColumn37,
			XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNy, 78,
			XmNheight, 32,
			NULL );


	/* Creation of optionMenu_p9 */
	optionMenu_p9_shell = XtVaCreatePopupShell ("optionMenu_p9_shell",
			xmMenuShellWidgetClass, form30,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p9 = XtVaCreateWidget( "optionMenu_p9",
			xmRowColumnWidgetClass,
			optionMenu_p9_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_p9_b2 */
	optionMenu_p9_b2 = XtVaCreateManagedWidget( "optionMenu_p9_b2",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "XYZT" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b2,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b3 */
	optionMenu_p9_b3 = XtVaCreateManagedWidget( "optionMenu_p9_b3",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "XYTZ" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b3, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b3,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b4 */
	optionMenu_p9_b4 = XtVaCreateManagedWidget( "optionMenu_p9_b4",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "XZYT" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b4, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b4,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b5 */
	optionMenu_p9_b5 = XtVaCreateManagedWidget( "optionMenu_p9_b5",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "XZTY" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b5, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b5,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b6 */
	optionMenu_p9_b6 = XtVaCreateManagedWidget( "optionMenu_p9_b6",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "XTZY" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b6, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b6,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b7 */
	optionMenu_p9_b7 = XtVaCreateManagedWidget( "optionMenu_p9_b7",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "XTYZ" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b7, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b7,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b8 */
	optionMenu_p9_b8 = XtVaCreateManagedWidget( "optionMenu_p9_b8",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "YXZT" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b8, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b8,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b9 */
	optionMenu_p9_b9 = XtVaCreateManagedWidget( "optionMenu_p9_b9",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "YXTZ" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b9, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b9,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b10 */
	optionMenu_p9_b10 = XtVaCreateManagedWidget( "optionMenu_p9_b10",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "YZXT" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b10, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b10,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b11 */
	optionMenu_p9_b11 = XtVaCreateManagedWidget( "optionMenu_p9_b11",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "YZTX" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b11, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b11,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b12 */
	optionMenu_p9_b12 = XtVaCreateManagedWidget( "optionMenu_p9_b12",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "YTXZ" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b12, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b12,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b13 */
	optionMenu_p9_b13 = XtVaCreateManagedWidget( "optionMenu_p9_b13",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "YTZX" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b13, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b13,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b14 */
	optionMenu_p9_b14 = XtVaCreateManagedWidget( "optionMenu_p9_b14",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "ZXYT" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b14, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b14,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b15 */
	optionMenu_p9_b15 = XtVaCreateManagedWidget( "optionMenu_p9_b15",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "ZXTY" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b15, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b15,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b16 */
	optionMenu_p9_b16 = XtVaCreateManagedWidget( "optionMenu_p9_b16",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "ZYZT" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b16, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b16,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b17 */
	optionMenu_p9_b17 = XtVaCreateManagedWidget( "optionMenu_p9_b17",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "ZYTZ" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b17, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b17,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b18 */
	optionMenu_p9_b18 = XtVaCreateManagedWidget( "optionMenu_p9_b18",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "ZTXY" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b18, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b18,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b19 */
	optionMenu_p9_b19 = XtVaCreateManagedWidget( "optionMenu_p9_b19",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "ZTYX" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b19, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b19,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b20 */
	optionMenu_p9_b20 = XtVaCreateManagedWidget( "optionMenu_p9_b20",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "TXYZ" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b20, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b20,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b21 */
	optionMenu_p9_b21 = XtVaCreateManagedWidget( "optionMenu_p9_b21",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "TXZY" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b21, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b21,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b22 */
	optionMenu_p9_b22 = XtVaCreateManagedWidget( "optionMenu_p9_b22",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "TYXZ" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b22, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b22,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b23 */
	optionMenu_p9_b23 = XtVaCreateManagedWidget( "optionMenu_p9_b23",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "TYZX" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b23, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b23,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b24 */
	optionMenu_p9_b24 = XtVaCreateManagedWidget( "optionMenu_p9_b24",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "TZXY" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b24, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b24,
		(XtPointer) NULL );



	/* Creation of optionMenu_p9_b26 */
	optionMenu_p9_b26 = XtVaCreateManagedWidget( "optionMenu_p9_b26",
			xmPushButtonWidgetClass,
			optionMenu_p9,
			RES_CONVERT( XmNlabelString, "TZYX" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( optionMenu_p9_b26, XmNactivateCallback,
		(XtCallbackProc) activateCB_optionMenu_p9_b26,
		(XtPointer) NULL );



	/* Creation of optionMenu17 */
	optionMenu17 = XtVaCreateManagedWidget( "optionMenu17",
			xmRowColumnWidgetClass,
			form30,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p9,
			XmNwidth, 86,
			XmNheight, 30,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNmarginWidth, 0,
			XmNspacing, 0,
			RES_CONVERT( XmNlabelString, " " ),
			XmNleftOffset, -11,
			XmNleftWidget, rowColumn37,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNbottomOffset, 80,
			XmNbottomWidget, rowColumn37,
			XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET,
			NULL );


	/* Creation of textField1 */
	textField1 = XtVaCreateManagedWidget( "textField1",
			xmTextFieldWidgetClass,
			form28,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 7,
			XmNleftWidget, rowColumn36,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 15,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );



	return ( SaveDataObject );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_SaveDataObject( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedSaveDataObject == NULL) {
		rtrn = _Uxbuild_SaveDataObject();

		}
		else
			rtrn = gSavedSaveDataObject;
				
		XtPopup(rtrn, no_grab);
		
		/* set the hi rez size */
		if (gHiRez && !gSavedSaveDataObject) {
			Dimension height, width;
				
			XtVaGetValues(form28,
				XmNheight, &height,
				XmNwidth, &width,
				NULL);
			width = 1.2 * width;
			height = 1.1 * height;
				
			XtVaSetValues(form28,
				XmNheight, height,
				XmNwidth, width,
				NULL);
		}
		
		if (gSavedSaveDataObject == NULL)
			gSavedSaveDataObject = rtrn;
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

