
/*******************************************************************************
	PrintSetup.c

       Associated Header file: PrintSetup.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/RowColumn.h>
#include <Xm/Frame.h>
#include <Xm/ToggleB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "ferret_structures.h"
#include "portrait.xpm"
#include "portrait_sel.xpm"
#include "landscape_sel.xpm"
#include "landscape.xpm"

#define DONT_UPDATE_MM 3

/* globals */
swidget PrintSetup, gSavedPrintSetup = NULL;
Boolean gPageSetupActive = False, gMetaCreationActive = False;
typedef struct prset {
	char outputDest;
	char printerOrFile[256];
	char orient;
	char numCopies[4];
	char lineStyle;
	char rename;
} printRec;

printRec gPrintSetup;
extern Boolean gHiRez;

/*protoypes */
swidget create_PrintSetup(swidget UxParent);
void ReadPrintSettings(void);
void SavePrintSettings(void);
void CreateDefaultPrintSettings(void);
void SettingsToInterface(void);
void InterfaceToSettings(void);
static void InitPixmaps();
extern void ferret_command(char *cmdText, int cmdMode);
extern void MaintainMainWdBtns(void);
extern Pixmap GetPixmapFromData(char **inData);
void GenPrintCmd(char *prnCmd);
void PrintCmdCB(void);
void InitPS(void);


static	Widget	form10;
static	Widget	toggleButton1;
static	Widget	frame9;
static	Widget	form16;
static	Widget	rowColumn1;
static	Widget	toggleButton3;
static	Widget	toggleButton4;
static	Widget	label25;
static	Widget	label26;
static	Widget	label48;
static	Widget	textField11;
static	Widget	textField12;
static	Widget	rowColumn2;
static	Widget	toggleButton5;
static	Widget	toggleButton6;
static	Widget	toggleButton36;
static	Widget	textField13;
static	Widget	rowColumn3;
static	Widget	toggleButton7;
static	Widget	toggleButton8;
static	Widget	rowColumn27;
static	Widget	pushButton9;
static	Widget	toggleButton2;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "PrintSetup.h"
#undef CONTEXT_MACRO_ACCESS

Widget	PrintSetup;
Widget	pushButton10;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

void InitPS()
{
	ReadPrintSettings(); 
	SettingsToInterface();
}

void PrintCmdCB()
{
	char prnCmd[80];

	strcpy(prnCmd, "SPAWN Fprint ");
	GenPrintCmd(prnCmd);
	strcat(prnCmd, "metafile.plt &");
	/* ferret_command(prnCmd,  DONT_UPDATE_MM); */
	ferret_command("PPL CLSPLT",  IGNORE_COMMAND_WIDGET);
	ferret_command(prnCmd,  IGNORE_COMMAND_WIDGET);
}

void GenPrintCmd(prnCmd)
char *prnCmd;
{
	char printer[80];
	int copies=0;

	if (gPrintSetup.outputDest == 'p') {
		/* route to printer */
		strcpy(printer, gPrintSetup.printerOrFile);
		if (strlen(printer) == 0) {
			strcpy(printer, getenv("PRINTER"));	/* default printer */
			if (strlen(printer) == 0) 
				strcpy(printer, "lp");
		}
		strcat(prnCmd, "-P ");
		strcat(prnCmd, printer);
		strcat(prnCmd, " ");
	}
	else {
		/* route to file */
		if (strlen(gPrintSetup.printerOrFile) == 0) {
			/* an error */
			return;
		}
		strcat(prnCmd, "-o ");
		strcat(prnCmd, gPrintSetup.printerOrFile);
		strcat(prnCmd, " ");
	}
		
	if (gPrintSetup.orient == 'p')
		strcat(prnCmd, "-p portrait ");
	else if (gPrintSetup.orient == 'l')
		strcat(prnCmd, "-p landscape ");

	sscanf(gPrintSetup.numCopies, "%d", &copies);
	if (copies > 1) {
		strcat(prnCmd, "-# ");
		strcat(prnCmd, gPrintSetup.numCopies);
		strcat(prnCmd, " ");
	}

	if (gPrintSetup.lineStyle == 'c')
		strcat(prnCmd, "-l cps ");
	else
		strcat(prnCmd, "-l ps ");

	strcat(prnCmd, "-R ");
}

static void InitPixmaps()
{
	XtVaSetValues(UxGetWidget(toggleButton6),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(portrait_xpm),
		XmNselectPixmap, GetPixmapFromData(portrait_sel_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton36),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(landscape_xpm),
		XmNselectPixmap, GetPixmapFromData(landscape_sel_xpm),
		NULL);
}

void SettingsToInterface()
{
	if (gPrintSetup.lineStyle == 'c')
		XmToggleButtonSetState(UxGetWidget(toggleButton7), True, True);
	else
		XmToggleButtonSetState(UxGetWidget(toggleButton8), True, True);

	XmTextFieldSetString(UxGetWidget(textField13), gPrintSetup.numCopies);

	if (gPrintSetup.orient == 'p')
		XmToggleButtonSetState(UxGetWidget(toggleButton6), True, True);
	else if (gPrintSetup.orient == 'l')
		XmToggleButtonSetState(UxGetWidget(toggleButton36), True, True);
	else 
		XmToggleButtonSetState(UxGetWidget(toggleButton5), True, True);
		
	if (gPrintSetup.outputDest == 'p') {
		XmTextFieldSetString(UxGetWidget(textField11), gPrintSetup.printerOrFile);
		XmToggleButtonSetState(UxGetWidget(toggleButton3), True, True);
	}
	else {
		XmTextFieldSetString(UxGetWidget(textField12), gPrintSetup.printerOrFile);
		XmToggleButtonSetState(UxGetWidget(toggleButton4), True, True);
	}
}

void InterfaceToSettings()
{
	if (XmToggleButtonGetState(UxGetWidget(toggleButton3))) {
		gPrintSetup.outputDest = 'p';	
		strcpy(gPrintSetup.printerOrFile, XmTextFieldGetString(UxGetWidget(textField11)));
		if (strlen(gPrintSetup.printerOrFile) == 0) 
			strcpy(gPrintSetup.printerOrFile, getenv("PRINTER"));
	}
	else {
		gPrintSetup.outputDest = 'f';
		strcpy(gPrintSetup.printerOrFile, XmTextFieldGetString(UxGetWidget(textField12)));
	}

	if (XmToggleButtonGetState(UxGetWidget(toggleButton5)))
		gPrintSetup.orient = 'a';
	else if (XmToggleButtonGetState(UxGetWidget(toggleButton6)))
		gPrintSetup.orient = 'p';
	else
		gPrintSetup.orient = 'l';

	strcpy(gPrintSetup.numCopies, XmTextFieldGetString(UxGetWidget(textField13)));
		if (strlen(gPrintSetup.numCopies) == 0) 
			strcpy(gPrintSetup.printerOrFile,"1");

	if (XmToggleButtonGetState(UxGetWidget(toggleButton7)))
		gPrintSetup.lineStyle = 'c';
	else
		gPrintSetup.lineStyle = 'p';
}

void ReadPrintSettings()
{
	char *home, homePath[256], buffer[512];
  	FILE *fp;

	/* save this to a file */
  	home = (char *)getenv("HOME");
 	if ( home ) {
    		strcpy(homePath, home);
    		strcat(homePath, "/.ferret.prec");
	}
	else 
    		strcpy(homePath, ".ferret.prec");
		
   	fp = fopen(homePath, "r");
    	if (fp != NULL) {
		/* print record exists--read the file */
		fscanf(fp, "%c\n", &gPrintSetup.outputDest);
		fscanf(fp, "%s\n", gPrintSetup.printerOrFile);
		fscanf(fp, "%c\n", &gPrintSetup.orient);
		fscanf(fp, "%s\n", gPrintSetup.numCopies);
		fscanf(fp, "%c\n", &gPrintSetup.lineStyle);
		fscanf(fp, "%c\n", &gPrintSetup.rename);
    		fclose(fp);
	}
	else {
		/* print record doesn't exist--create it */
		CreateDefaultPrintSettings();
		SavePrintSettings();
	}
}

void SavePrintSettings()
{
	char *home, homePath[256];
  	FILE *fp;

	/* save this to a file */
  	home = (char *)getenv("HOME");
 	if (home) {
    		strcpy(homePath, home);
    		strcat(homePath, "/.ferret.prec");
	}
	else 
		/* look in the current directory */
    		strcpy(homePath, ".ferret.prec");

   	fp = fopen(homePath, "w");
    	if (fp != NULL) {
		/* write the file */
		fprintf(fp, "%c\n", gPrintSetup.outputDest);
		fprintf(fp, "%s\n", gPrintSetup.printerOrFile);
		fprintf(fp, "%c\n", gPrintSetup.orient);
		fprintf(fp, "%s\n", gPrintSetup.numCopies);
		fprintf(fp, "%c\n", gPrintSetup.lineStyle);
		fprintf(fp, "%c\n", gPrintSetup.rename);
    		fclose(fp);
	}
}

void CreateDefaultPrintSettings()
{
	char *printer;

	gPrintSetup.outputDest = 'p';
	/* output is printer */
	printer = (char *)getenv("PRINTER");
	if (printer)
		strcpy(gPrintSetup.printerOrFile, printer);	/* default printer */
	else
		strcpy(gPrintSetup.printerOrFile, "Default Printer Undefined");
	gPrintSetup.orient = 'a';				/* auto */
	strcpy(gPrintSetup.numCopies, "1");			/* num copies */
	gPrintSetup.lineStyle = 'c';				/* line style */
	gPrintSetup.rename = 'y';

	/* write the file */
	SavePrintSettings();
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	destroyCB_PrintSetup(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedPrintSetup = NULL;
}

static	void	valueChangedCB_toggleButton3(
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
		XtMapWidget(UxGetWidget(textField11));
		XtUnmapWidget(UxGetWidget(textField12));
	}
	
	}
}

static	void	valueChangedCB_toggleButton4(
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
		XtMapWidget(UxGetWidget(textField12));
		XtUnmapWidget(UxGetWidget(textField11));
	}
	
	}
}

static	void	valueChangedCB_toggleButton5(
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

static	void	valueChangedCB_toggleButton6(
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

static	void	valueChangedCB_toggleButton7(
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

static	void	valueChangedCB_toggleButton8(
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

static	void	activateCB_pushButton9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XtPopdown(UxGetWidget(PrintSetup));
	}
}

static	void	activateCB_pushButton10(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	InterfaceToSettings();
	SavePrintSettings();
	
	/* test how to set metafile mode */
	
	/* need to query about current mode here */
	
	if (XmToggleButtonGetState(UxGetWidget(toggleButton2)) && !gMetaCreationActive) {
		gMetaCreationActive = True;
		ferret_command("SET MODE METAFILE", IGNORE_COMMAND_WIDGET);
	}
	else if (!XmToggleButtonGetState(UxGetWidget(toggleButton2)) && gMetaCreationActive) {
		gMetaCreationActive = False;
		ferret_command("CANCEL MODE METAFILE", IGNORE_COMMAND_WIDGET);
	}
	MaintainMainWdBtns();
	
	XtPopdown(UxGetWidget(PrintSetup));
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_PrintSetup()
{
	Widget		_UxParent;


	/* Creation of PrintSetup */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	PrintSetup = XtVaCreatePopupShell( "PrintSetup",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 359,
			XmNheight, 349,
			XmNx, 0,
			XmNy, 0,
			XmNiconName, "Ferret: Print Setup",
			XmNtitle, "Ferret: Print Setup",
			NULL );
	XtAddCallback( PrintSetup, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_PrintSetup,
		(XtPointer) NULL );



	/* Creation of form10 */
	form10 = XtVaCreateManagedWidget( "form10",
			xmFormWidgetClass,
			PrintSetup,
			XmNresizePolicy, XmRESIZE_GROW,
			XmNunitType, XmPIXELS,
			XmNwidth, 359,
			XmNheight, 349,
			NULL );


	/* Creation of toggleButton1 */
	toggleButton1 = XtVaCreateManagedWidget( "toggleButton1",
			xmToggleButtonWidgetClass,
			form10,
			XmNx, 1784,
			XmNy, 1189,
			XmNwidth, 30934,
			XmNheight, 2280,
			RES_CONVERT( XmNlabelString, "Save Graphic Metafiles for Printing" ),
			NULL );


	/* Creation of frame9 */
	frame9 = XtVaCreateManagedWidget( "frame9",
			xmFrameWidgetClass,
			form10,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNbottomOffset, 50,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNtopOffset, 40,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of form16 */
	form16 = XtVaCreateManagedWidget( "form16",
			xmFormWidgetClass,
			frame9,
			XmNresizePolicy, XmRESIZE_GROW,
			XmNx, 2,
			XmNwidth, 337,
			XmNheight, 219,
			NULL );


	/* Creation of rowColumn1 */
	rowColumn1 = XtVaCreateManagedWidget( "rowColumn1",
			xmRowColumnWidgetClass,
			form16,
			XmNorientation, XmVERTICAL,
			XmNradioBehavior, TRUE,
			XmNpacking, XmPACK_TIGHT,
			XmNspacing, 8,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 7,
			XmNtopAttachment, XmATTACH_FORM,
			XmNx, 10,
			XmNwidth, 825,
			NULL );


	/* Creation of toggleButton3 */
	toggleButton3 = XtVaCreateManagedWidget( "toggleButton3",
			xmToggleButtonWidgetClass,
			rowColumn1,
			RES_CONVERT( XmNlabelString, "Printer:" ),
			NULL );
	XtAddCallback( toggleButton3, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton3,
		(XtPointer) NULL );



	/* Creation of toggleButton4 */
	toggleButton4 = XtVaCreateManagedWidget( "toggleButton4",
			xmToggleButtonWidgetClass,
			rowColumn1,
			RES_CONVERT( XmNlabelString, "PostScript File:" ),
			NULL );
	XtAddCallback( toggleButton4, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton4,
		(XtPointer) NULL );



	/* Creation of label25 */
	label25 = XtVaCreateManagedWidget( "label25",
			xmLabelWidgetClass,
			form16,
			RES_CONVERT( XmNlabelString, "Orientation:" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 13,
			XmNtopWidget, rowColumn1,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of label26 */
	label26 = XtVaCreateManagedWidget( "label26",
			xmLabelWidgetClass,
			form16,
			RES_CONVERT( XmNlabelString, "Copies:" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 19,
			XmNtopWidget, label25,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of label48 */
	label48 = XtVaCreateManagedWidget( "label48",
			xmLabelWidgetClass,
			form16,
			RES_CONVERT( XmNlabelString, "Line Style:" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 27,
			XmNtopWidget, label26,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of textField11 */
	textField11 = XtVaCreateManagedWidget( "textField11",
			xmTextFieldWidgetClass,
			form16,
			XmNsensitive, TRUE,
			XmNleftOffset, 1,
			XmNleftWidget, rowColumn1,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 4,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of textField12 */
	textField12 = XtVaCreateManagedWidget( "textField12",
			xmTextFieldWidgetClass,
			form16,
			XmNsensitive, TRUE,
			XmNleftOffset, 1,
			XmNleftWidget, rowColumn1,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 2,
			XmNtopWidget, textField11,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of rowColumn2 */
	rowColumn2 = XtVaCreateManagedWidget( "rowColumn2",
			xmRowColumnWidgetClass,
			form16,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNpacking, XmPACK_TIGHT,
			XmNleftOffset, 2,
			XmNleftWidget, label25,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 4,
			XmNtopWidget, textField12,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of toggleButton5 */
	toggleButton5 = XtVaCreateManagedWidget( "toggleButton5",
			xmToggleButtonWidgetClass,
			rowColumn2,
			RES_CONVERT( XmNlabelString, "Auto" ),
			XmNlabelType, XmSTRING,
			NULL );
	XtAddCallback( toggleButton5, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton5,
		(XtPointer) NULL );



	/* Creation of toggleButton6 */
	toggleButton6 = XtVaCreateManagedWidget( "toggleButton6",
			xmToggleButtonWidgetClass,
			rowColumn2,
			RES_CONVERT( XmNlabelString, "portrait" ),
			XmNlabelType, XmPIXMAP,
			NULL );
	XtAddCallback( toggleButton6, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton6,
		(XtPointer) NULL );



	/* Creation of toggleButton36 */
	toggleButton36 = XtVaCreateManagedWidget( "toggleButton36",
			xmToggleButtonWidgetClass,
			rowColumn2,
			RES_CONVERT( XmNlabelString, "landscape" ),
			XmNlabelType, XmPIXMAP,
			NULL );


	/* Creation of textField13 */
	textField13 = XtVaCreateManagedWidget( "textField13",
			xmTextFieldWidgetClass,
			form16,
			XmNwidth, 51,
			XmNsensitive, TRUE,
			XmNleftOffset, 5,
			XmNleftWidget, label26,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 1,
			XmNtopWidget, rowColumn2,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of rowColumn3 */
	rowColumn3 = XtVaCreateManagedWidget( "rowColumn3",
			xmRowColumnWidgetClass,
			form16,
			XmNorientation, XmVERTICAL,
			XmNradioBehavior, TRUE,
			XmNpacking, XmPACK_TIGHT,
			XmNspacing, 0,
			XmNleftOffset, 0,
			XmNleftWidget, label48,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 2,
			XmNtopWidget, textField13,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of toggleButton7 */
	toggleButton7 = XtVaCreateManagedWidget( "toggleButton7",
			xmToggleButtonWidgetClass,
			rowColumn3,
			RES_CONVERT( XmNlabelString, "Color Lines (cps)" ),
			NULL );
	XtAddCallback( toggleButton7, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton7,
		(XtPointer) NULL );



	/* Creation of toggleButton8 */
	toggleButton8 = XtVaCreateManagedWidget( "toggleButton8",
			xmToggleButtonWidgetClass,
			rowColumn3,
			RES_CONVERT( XmNlabelString, "Dot-Dashed Lines (ps)" ),
			NULL );
	XtAddCallback( toggleButton8, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton8,
		(XtPointer) NULL );



	/* Creation of rowColumn27 */
	rowColumn27 = XtVaCreateManagedWidget( "rowColumn27",
			xmRowColumnWidgetClass,
			form10,
			XmNentryAlignment, XmALIGNMENT_CENTER,
			XmNorientation, XmHORIZONTAL,
			XmNpacking, XmPACK_COLUMN,
			XmNleftPosition, 29,
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 10,
			XmNtopWidget, frame9,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNx, 107,
			XmNy, 274,
			XmNspacing, 3,
			NULL );


	/* Creation of pushButton9 */
	pushButton9 = XtVaCreateManagedWidget( "pushButton9",
			xmPushButtonWidgetClass,
			rowColumn27,
			RES_CONVERT( XmNlabelString, "Cancel" ),
			XmNx, -5,
			XmNy, 6,
			NULL );
	XtAddCallback( pushButton9, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton9,
		(XtPointer) NULL );



	/* Creation of pushButton10 */
	pushButton10 = XtVaCreateManagedWidget( "pushButton10",
			xmPushButtonWidgetClass,
			rowColumn27,
			RES_CONVERT( XmNlabelString, "OK" ),
			XmNx, 29,
			XmNy, 3,
			NULL );
	XtAddCallback( pushButton10, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton10,
		(XtPointer) NULL );



	/* Creation of toggleButton2 */
	toggleButton2 = XtVaCreateManagedWidget( "toggleButton2",
			xmToggleButtonWidgetClass,
			form10,
			RES_CONVERT( XmNlabelString, "Enable Printing" ),
			XmNleftOffset, 11,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );



	return ( PrintSetup );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_PrintSetup( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedPrintSetup == NULL) {
		rtrn = _Uxbuild_PrintSetup();

		/* install the pixmaps */
			InitPixmaps();
		}
		else
			rtrn = gSavedPrintSetup;
				
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		/* set the hi rez size */
		if (gHiRez && !gSavedPrintSetup) {
			Dimension height, width;
				
			XtVaGetValues(UxGetWidget(form10),
				XmNheight, &height,
				XmNwidth, &width,
				NULL);
			width = 1.2 * width;
			height = 1.1 * height;
				
			XtVaSetValues(UxGetWidget(form10),
				XmNheight, height,
				XmNwidth, width,
				NULL);
		}
		
		if (gSavedPrintSetup == NULL)
			gSavedPrintSetup = rtrn;
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

