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
	MacroManager.c

       Associated Header file: MacroManager.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/Label.h>
#include <Xm/ToggleB.h>
#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/CascadeB.h>
#include <Xm/Separator.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <Xm/PushB.h>
#include "ferret_structures.h"
#include "mm_run_out_ins.xpm"
#include "mm_run_out.xpm"
#include "mm_run_in.xpm"
#include "mm_rec_in.xpm"
#include "mm_rec_out.xpm"
#include "mm_rec_out_ins.xpm"
#include "mm_stop_out_ins.xpm"
#include "mm_stop_in.xpm"
#include "mm_stop_out.xpm"

#define YES 1
#define NO 2
#define CANCEL 3
#define DONT_UPDATE_MM 3

/* globals */
char *macroBuffer;
extern int gMacroIsRecording, gMMIsOpen;
swidget gSavedMacroManager = NULL;
extern swidget Save_jnl, fileSelectionBox4, CommandHelp;
extern swidget Open_jnl, fileSelectionBox5;
swidget MacroManager;
static int macroIsDirty = 0;
static int macroHasText = 0;
static int macroSavedOnce = 0;
static char defaultPath[256];
static Boolean iOwnClip = 0;
int mmPos=0;

/* prototypes */
swidget create_MacroManager(swidget UxParent);
extern swidget create_Open_jnl(swidget UxParent);
extern swidget create_Save_jnl(swidget UxParent);
extern swidget create_CommandHelp(swidget UxParent);
static void InitialState(void);
static void ClearRadioGroup1(void);
static void MaintainButtons(void);
void MMSaveOK(void);
void MMSaveAsOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
void MMOpenOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
void MMCancelSave(void);
void MMCancelOpen(void);
static void TextChangedCB(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
void SaveAsMMFile(void);
void OpenMMFile(void);
void DisplayMacroBuffer(void);
void SetRecordBtn(void);
void SetStopBtn(void);
extern void MaintainMainMenu();
extern void ferret_command(char *cmdText, int cmdMode);
extern char *CollectToReturn(char *targetStr, char *subStr);
void RunMacro(void);
int AskUser(Widget parent, char *question, char *ans1, char *ans2, int default_ans);
void response(Widget widget, XtPointer client_data, XtPointer call_data);
int AskUser2(Widget parent, char *question, char *ans1, char *ans2, int default_ans);
extern Pixmap GetPixmapFromData(char **inData);
static void InitPixmaps();


static	Widget	form12;
static	Widget	menuBar1;
static	Widget	menuBar_p1;
static	Widget	menuBar_p_b1;
static	Widget	menuBar_p1_b3;
static	Widget	menuBar_p1_b2;
static	Widget	menuBar_p1_b4;
static	Widget	menuBar_p1_b5;
static	Widget	menuBar_p1_b6;
static	Widget	menuBar_top_b2;
static	Widget	menuBar1_p2;
static	Widget	menuBar1_p2_b1;
static	Widget	menuBar1_top_b3;
static	Widget	menuBar1_p3;
static	Widget	menuBar1_p3_b1;
static	Widget	menuBar1_top_b5;
static	Widget	scrolledWindowText1;
static	Widget	scrolledText1;
static	Widget	rowColumn18;
static	Widget	toggleButton68;
static	Widget	toggleButton69;
static	Widget	toggleButton70;
static	Widget	label60;
static	Widget	label61;
static	Widget	label62;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "MacroManager.h"
#undef CONTEXT_MACRO_ACCESS

Widget	MacroManager;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static void InitPixmaps()
{
	XtVaSetValues(UxGetWidget(toggleButton68),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(mm_rec_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(mm_rec_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(mm_rec_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton69),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(mm_stop_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(mm_stop_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(mm_stop_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton70),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(mm_run_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(mm_run_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(mm_run_in_xpm),
		NULL);
}

/*
 * AskUser() -- a generalized routine that asks the user a question
 * and returns a response.  Parameters are: the question, the labels
 * for the "Yes" and "No" buttons, and the default selection to use.
 */
int AskUser(parent, question, ans1, ans2, default_ans)
Widget parent;
char *question, *ans1, *ans2;
int default_ans;
{
    int n = 0;
    Widget noButton;
    static Widget dialog; /* static to avoid multiple creation */
    XmString text, yes, no;
    static int answer;
    Display *dis;
    XtAppContext app;
    ArgList args;

    args = (ArgList)malloc(5 * sizeof(Arg));
    answer = 0;
    text = XmStringCreateLocalized (question);
    yes = XmStringCreateLocalized (ans1);
    no = XmStringCreateLocalized (ans2);

    if (!dialog) {
	/* customize the dialog */
	XtSetArg(args[n], XmNautoUnmanage, False); n++;
	XtSetArg(args[n], XmNmessageString, text); n++;
	XtSetArg(args[n], XmNokLabelString, yes); n++;

       dialog = (Widget)XmCreateQuestionDialog(parent, "Macro Manager Alert", args, n);

	noButton = XtVaCreateManagedWidget("no", 
		xmPushButtonWidgetClass, dialog,
		XmNlabelString, no,
		NULL);

        XtVaSetValues (dialog,
            XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL,
            NULL);
        XtUnmanageChild((Widget)XmMessageBoxGetChild (dialog, XmDIALOG_HELP_BUTTON));

        XtAddCallback (dialog, XmNokCallback, response, &answer);
        XtAddCallback (noButton, XmNactivateCallback, response, &answer);
        XtAddCallback(dialog, XmNcancelCallback, response, &answer);
    }

     XtVaSetValues (dialog,
       	 XmNdefaultButtonType,  default_ans == YES ?
       	 XmDIALOG_OK_BUTTON : XmDIALOG_CANCEL_BUTTON,
         NULL);

    dis = XtDisplayOfObject(dialog);
    app = XtDisplayToApplicationContext(dis);

    XmStringFree (text);
    XmStringFree (yes);
    XmStringFree (no);
    XtManageChild (dialog);
    XtPopup (XtParent (dialog), XtGrabNone);

    while (answer == 0)
        XtAppProcessEvent (app, XtIMAll);

    XtPopdown (XtParent (dialog));
    /* make sure the dialog goes away before returning. Sync with server
     * and update the display.
     */
    XSync (XtDisplay (dialog), 0);
    XmUpdateDisplay (parent);

    return answer;
}

int AskUser2(parent, question, ans1, ans2, default_ans)
Widget parent;
char *question, *ans1, *ans2;
int default_ans;
{
    int n = 0;
    Widget noButton;
    static Widget dialog; /* static to avoid multiple creation */
    XmString text, yes, no;
    static int answer;
    Display *dis;
    XtAppContext app;
    Arg args[5];

    answer = 0;
    text = XmStringCreateLocalized (question);
    yes = XmStringCreateLocalized (ans1);
    no = XmStringCreateLocalized (ans2);

    if (!dialog) {
	/* customize the dialog */
	XtSetArg(args[n], XmNautoUnmanage, False); n++;
	XtSetArg(args[n], XmNmessageString, text); n++;
	XtSetArg(args[n], XmNokLabelString, yes); n++;

        dialog = (Widget)XmCreateQuestionDialog (parent, "Macro Manager Alert", args, n);
	noButton = XtVaCreateManagedWidget("no", 
		xmPushButtonWidgetClass, dialog,
		XmNlabelString, no,
		NULL);

        XtVaSetValues (dialog,
            XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL,
            NULL);
        XtUnmanageChild((Widget)XmMessageBoxGetChild (dialog, XmDIALOG_HELP_BUTTON));

	XtUnmanageChild((Widget)XmMessageBoxGetChild (dialog, XmDIALOG_CANCEL_BUTTON));
        XtAddCallback (dialog, XmNokCallback, response, &answer);
        XtAddCallback (noButton, XmNactivateCallback, response, &answer);
    }

     XtVaSetValues (dialog,
       	 XmNdefaultButtonType,  default_ans == YES ?
       	 XmDIALOG_OK_BUTTON : XmDIALOG_CANCEL_BUTTON,
         NULL);

    dis = XtDisplayOfObject(dialog);
    app = XtDisplayToApplicationContext(dis);

    XmStringFree (text);
    XmStringFree (yes);
    XmStringFree (no);
    XtManageChild (dialog);
    XtPopup (XtParent (dialog), XtGrabNone);

    while (answer == 0)
        XtAppProcessEvent (app, XtIMAll);

    XtPopdown (XtParent (dialog));
    /* make sure the dialog goes away before returning. Sync with server
     * and update the display.
     */
    XSync (XtDisplay (dialog), 0);
    XmUpdateDisplay (parent);

    return answer;
}
/* response() --The user made some sort of response to the
 * question posed in AskUser().  Set the answer (client_data)
 * accordingly.
 */
void response(widget, client_data, call_data)
Widget widget;
XtPointer client_data;
XtPointer call_data;
{
    int *answer = (int *) client_data;
    XmAnyCallbackStruct *cbs = (XmAnyCallbackStruct *) call_data;

    if (cbs->reason == XmCR_OK)
        *answer = YES;
    else if (cbs->reason == XmCR_CANCEL)
        *answer = CANCEL;
    else if (cbs->reason == XmCR_ACTIVATE)
        *answer = NO;
}

static void InitialState()
{
	/* set record/stop/run btns according to current recording state */
	if (gMacroIsRecording)
		SetRecordBtn();
	else 
		SetStopBtn();

	/* display macrobuffer */
	DisplayMacroBuffer();
	MaintainButtons();
	gMMIsOpen = 1;
}

void SetRecordBtn()
{
	ClearRadioGroup1();
	XtVaSetValues(UxGetWidget(toggleButton68),
		XmNset, True,
		NULL);
	XtUnmapWidget(UxGetWidget(rowColumn18));
	XtMapWidget(UxGetWidget(rowColumn18));
	MaintainButtons();
}

void SetStopBtn()
{
	ClearRadioGroup1();
	XtVaSetValues(UxGetWidget(toggleButton69),
		XmNset, True,
		NULL);
	XtUnmapWidget(UxGetWidget(rowColumn18));
	XtMapWidget(UxGetWidget(rowColumn18));
	MaintainButtons();
}

static void ClearRadioGroup1()	
{
	XtVaSetValues(UxGetWidget(toggleButton68),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton69),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton70),
		XmNset, False,
		NULL);
}

void DisplayMacroBuffer()
{
	if (mmPos = strlen(macroBuffer)) {
		XtVaSetValues(UxGetWidget(scrolledText1),
			XmNvalue, macroBuffer,
			NULL);
		XtVaSetValues(UxGetWidget(scrolledText1),
			XmNcursorPosition, mmPos,
			NULL);
		XmTextShowPosition(UxGetWidget(scrolledText1), mmPos);
		macroHasText = 1;
		macroIsDirty = 1;
	}
	else
		macroHasText = 0;
	MaintainButtons();
}

static void MaintainButtons()
{
	if (macroIsDirty && macroHasText && macroSavedOnce)
		/* can save */
		XtSetSensitive(UxGetWidget(menuBar_p1_b2), True);
	else
		XtSetSensitive(UxGetWidget(menuBar_p1_b2), False);

	if (macroIsDirty && macroHasText)
		/* can save as */
		XtSetSensitive(UxGetWidget(menuBar_p1_b4), True);
	else
		XtSetSensitive(UxGetWidget(menuBar_p1_b4), False);

	if (macroHasText)
		/* can clear */
		XtSetSensitive(UxGetWidget(menuBar1_p2_b1), True);
	else
		/* can't clear */
		XtSetSensitive(UxGetWidget(menuBar1_p2_b1), False);

	if (macroHasText)
		/* can run */
		XtSetSensitive(UxGetWidget(toggleButton70), True);
	else
		/* can't run */
		XtSetSensitive(UxGetWidget(toggleButton70), False);
}

/* ok and cancel callbacks for fileSelectionBox1 */

extern void MMSaveAsOK(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	XmSelectionBoxCallbackStruct *cbInfo = (XmSelectionBoxCallbackStruct *)UxCallbackArg;
	char *pathName, *contents;
	FILE *outFile;
	int io;
	struct stat buf;
	int defaultBtn=NO, result;
	char *question={"File already exists. Overwrite?"};
	char *yes={"Yes"};
	char *no={"No"};

	pathName = (char *)XtMalloc(cbInfo->length);
	XmStringGetLtoR(cbInfo->value, XmSTRING_DEFAULT_CHARSET, &pathName);

	/* test whether file exists or is writable */
	if (stat(pathName, &buf) == 0) {
		/* file already exists --ask to overwrite */
		result = AskUser2(UxGetWidget(MacroManager), question, yes, no, defaultBtn);
		if (result == NO)
			return;	
	}

	/* open the file */
	outFile = fopen(pathName, "w");
	strcpy(defaultPath, pathName);
	
	/* get a pointer to contour text */
	contents = (char *)XtMalloc(5000);
	XtVaGetValues(UxGetWidget(scrolledText1),
		XmNvalue, &contents,
		NULL);	

	/* write text to file */
	io = fwrite(contents, sizeof(char), strlen(contents), outFile);

	/* close file */
	io = fclose(outFile);

	/* pop down the interface */
	XtPopdown(UxGetWidget(Save_jnl));

	XtFree(pathName); /* allocated with XtMalloc() */
	XtFree(contents); /* allocated with XtMalloc() */

	macroIsDirty = 0;
	macroSavedOnce = 1;
	MaintainButtons();
}

extern void MMSaveOK()
{
	char *contents;
	FILE *outFile;
	int io;

	/* open the file */
	outFile = fopen(defaultPath, "w");
	
	/* get a pointer to contour text */
	contents = (char *)XtMalloc(5000);
	XtVaGetValues(UxGetWidget(scrolledText1),
		XmNvalue, &contents,
		NULL);	

	/* write text to file */
	io = fwrite(contents, sizeof(char), strlen(contents), outFile);

	/* close file */
	io = fclose(outFile);

	/* pop down the interface */
	XtDestroyWidget(UxGetWidget(Save_jnl));

	XtFree(contents); /* allocated with XtMalloc() */

	macroIsDirty = 0;
	MaintainButtons();
}

extern void MMOpenOK(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	XmSelectionBoxCallbackStruct *cbInfo = (XmSelectionBoxCallbackStruct *)UxCallbackArg;
	char *pathName, *fileContents, *currContents;
	FILE *inFile;
	int io;

	XmStringGetLtoR(cbInfo->value, XmSTRING_DEFAULT_CHARSET, &pathName);

	/* open the file */
	inFile = fopen(pathName, "r");
	
	/* create a buffer to store contour text */
	fileContents = (char *)XtMalloc(50000);
	strcpy(fileContents, "");

	/* read text from file */
	io = fread(fileContents, sizeof(char), 50000, inFile);

	/* close file */
	io = fclose(inFile); 

	/* append this the contents of field */
	currContents = (char *)XtMalloc(32000);
	strcpy(currContents, macroBuffer);
	strcat(currContents, fileContents);
	strcpy(macroBuffer, currContents);

	macroIsDirty = 1;
	
	DisplayMacroBuffer();

	/* pop down the interface */
	XtDestroyWidget(UxGetWidget(Open_jnl));

	XtFree(pathName); /* allocated with XmStringGetLtoR() */
	XtFree(fileContents); /* allocated with XtMalloc() */
	XtFree(currContents); /* allocated with XtMalloc() */
}

void MMCancelSave()
{
	/* pop down the interface */
	XtDestroyWidget(UxGetWidget(Save_jnl));
}

void MMCancelOpen()
{
	/* pop down the interface */
	XtDestroyWidget(UxGetWidget(Open_jnl));
}

void SaveAsMMFile()
{
	XmString mask, selection, dir;

	XtVaGetValues(UxGetWidget(fileSelectionBox4),
		XmNdirectory, &dir,
		NULL);

	mask = XmStringCreateLocalized("*.jnl");
	XtVaSetValues(UxGetWidget(fileSelectionBox4),
		XmNdirMask, mask,
		NULL);

	selection = XmStringConcat(dir, XmStringCreateLocalized(".jnl"));
	XtVaSetValues(UxGetWidget(fileSelectionBox4),
		XmNtextString, selection,
		NULL);

	/* popup Open file */
	XtPopup(UxGetWidget(Save_jnl), XtGrabNone);
	XmStringFree(mask); /* allocated with XmStringCreateLocalized() */
	XmStringFree(selection); /* allocated with XmStringConcat() */
	XmStringFree(dir); /* TODO: is this ever allocated?? */
}

void OpenMMFile()
{
	XmString mask;

	/* the open file window has been managed but is unmapped at this point */

	mask = XmStringCreateLocalized("*.jnl");
	XtVaSetValues(UxGetWidget(fileSelectionBox5),
		XmNdirMask, mask,
		NULL); 

	/* popup Open file */
	XtPopup(UxGetWidget(Open_jnl), XtGrabNone);
	XmStringFree(mask);
}

static void TextChangedCB(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	char *contents;

	/* replace macroBuffer with text field contents */
	contents = (char *)XtMalloc(32000);
	XtVaGetValues(UxGetWidget(scrolledText1),
		XmNvalue, &contents,
		NULL);
	strcpy(macroBuffer, contents);
	macroIsDirty = 1;
	if (strlen(macroBuffer))
		macroHasText = 1;

	XtFree(contents); /* allocated with XtMalloc() */
	MaintainButtons();
}

void RunMacro()
{
	char aCmnd[256], *startBuffer;
	int result;
	char *currSelection, *currSelection_ptr;

	/* disable the record and stop btns */
	XtSetSensitive(UxGetWidget(toggleButton68), False);
	XtSetSensitive(UxGetWidget(toggleButton69), False);

	if (iOwnClip) {
		/* process a selection */
		currSelection = XmTextGetSelection(UxGetWidget(scrolledText1));
		currSelection_ptr = currSelection;
		if (currSelection_ptr) {
			while (*currSelection_ptr) {
				currSelection_ptr = CollectToReturn(currSelection_ptr, (char *)aCmnd);
	
				/* send text to ferret command and check result code */
				ferret_command(aCmnd, DONT_UPDATE_MM);
			}
		}
		XtFree(currSelection); /* allocated with XmTextGetSelection() */
	}
	else {
		/*  get the macro text and parse it line by line */
		startBuffer = macroBuffer;

		while (*macroBuffer) {
			macroBuffer = CollectToReturn(macroBuffer, (char *)aCmnd);
	
			/* send text to ferret command and check result code */
			ferret_command(aCmnd, DONT_UPDATE_MM);
		}
		macroBuffer = startBuffer;
	}

	/* Turn on the record and stop buttons */
	XtSetSensitive(UxGetWidget(toggleButton68), True);
	XtSetSensitive(UxGetWidget(toggleButton69), True);

	if (gMacroIsRecording)
		SetRecordBtn();
	else
		SetStopBtn();
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	popupCB_MacroManager(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	InitialState();
}

static	void	destroyCB_MacroManager(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gMMIsOpen = 0;
}

static	void	popdownCB_MacroManager(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gMMIsOpen = 0;
}

static	void	destroyCB_form12(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedMacroManager = NULL;
}

static	void	activateCB_menuBar_p_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	Open_jnl = create_Open_jnl(NO_PARENT);
	OpenMMFile();
	MaintainButtons();
	}
}

static	void	activateCB_menuBar_p1_b2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	MMSaveOK();
	}
}

static	void	activateCB_menuBar_p1_b4(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	Save_jnl = create_Save_jnl(NO_PARENT);
	SaveAsMMFile();
}

static	void	activateCB_menuBar_p1_b6(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	XtPopdown(UxGetWidget(MacroManager));
}

static	void	activateCB_menuBar1_p2_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	int defaultBtn=YES, result;
	char *question={"Save macro before clearing text?"};
	char *yes={"Yes"};
	char *no={"No"};
	
	/* alert before clearing */
	if (macroIsDirty) {
		result = AskUser(UxGetWidget(MacroManager), question, yes, no, defaultBtn);
		if (result == YES) {
			if (macroSavedOnce)
				MMSaveOK();
			else {
				Save_jnl = create_Save_jnl(NO_PARENT);
				SaveAsMMFile();
			}
		}
	}
	
	if (result != CANCEL) {
		strcpy(macroBuffer, "");
		XtVaSetValues(UxGetWidget(scrolledText1),
			XmNvalue, macroBuffer,
			NULL);
		mmPos = 0;
		macroHasText = 0;
		macroIsDirty = 0;
		MaintainButtons();
	}
	}
}

static	void	activateCB_menuBar1_p3_b1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{extern swidget CommandHelp;
	
	CommandHelp = create_CommandHelp(NO_PARENT);}
}

static	void	valueChangedCB_scrolledText1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	TextChangedCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	modifyVerifyCB_scrolledText1(
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

static	void	motionVerifyCB_scrolledText1(
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

static	void	gainPrimaryCB_scrolledText1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	iOwnClip = True;
}

static	void	losePrimaryCB_scrolledText1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	iOwnClip = False;
}

static	void	valueChangedCB_toggleButton68(
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
		gMacroIsRecording = 1;
		MaintainMainMenu();
		MaintainButtons();
	}
	}
}

static	void	valueChangedCB_toggleButton69(
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
		gMacroIsRecording = 0;
		MaintainMainMenu();
		MaintainButtons();
	}
	
	}
}

static	void	valueChangedCB_toggleButton70(
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
		RunMacro();
	}
	
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_MacroManager()
{
	Widget		_UxParent;
	Widget		menuBar_p1_shell;
	Widget		menuBar1_p2_shell;
	Widget		menuBar1_p3_shell;


	/* Creation of MacroManager */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	MacroManager = XtVaCreatePopupShell( "MacroManager",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 707,
			XmNheight, 281,
			XmNx, 158,
			XmNy, 111,
			XmNiconName, "Ferret: Command Line Interface",
			XmNtitle, "Command Line Interface",
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( MacroManager, XmNpopupCallback,
		(XtCallbackProc) popupCB_MacroManager,
		(XtPointer) NULL );
	XtAddCallback( MacroManager, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_MacroManager,
		(XtPointer) NULL );
	XtAddCallback( MacroManager, XmNpopdownCallback,
		(XtCallbackProc) popdownCB_MacroManager,
		(XtPointer) NULL );



	/* Creation of form12 */
	form12 = XtVaCreateManagedWidget( "form12",
			xmFormWidgetClass,
			MacroManager,
			XmNwidth, 707,
			XmNheight, 281,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 33,
			XmNy, 30,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( form12, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_form12,
		(XtPointer) NULL );



	/* Creation of menuBar1 */
	menuBar1 = XtVaCreateManagedWidget( "menuBar1",
			xmRowColumnWidgetClass,
			form12,
			XmNrowColumnType, XmMENU_BAR,
			XmNx, 12,
			XmNy, 11,
			XmNwidth, 297,
			XmNheight, 31,
			XmNmenuAccelerator, "<KeyUp>F10",
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 0,
			XmNtopAttachment, XmATTACH_FORM,
			XmNrightOffset, 0,
			XmNrightAttachment, XmATTACH_FORM,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of menuBar_p1 */
	menuBar_p1_shell = XtVaCreatePopupShell ("menuBar_p1_shell",
			xmMenuShellWidgetClass, menuBar1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	menuBar_p1 = XtVaCreateWidget( "menuBar_p1",
			xmRowColumnWidgetClass,
			menuBar_p1_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of menuBar_p_b1 */
	menuBar_p_b1 = XtVaCreateManagedWidget( "menuBar_p_b1",
			xmPushButtonWidgetClass,
			menuBar_p1,
			RES_CONVERT( XmNlabelString, "Open..." ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( menuBar_p_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_menuBar_p_b1,
		(XtPointer) NULL );



	/* Creation of menuBar_p1_b3 */
	menuBar_p1_b3 = XtVaCreateManagedWidget( "menuBar_p1_b3",
			xmSeparatorWidgetClass,
			menuBar_p1,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of menuBar_p1_b2 */
	menuBar_p1_b2 = XtVaCreateManagedWidget( "menuBar_p1_b2",
			xmPushButtonWidgetClass,
			menuBar_p1,
			RES_CONVERT( XmNlabelString, "Save" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( menuBar_p1_b2, XmNactivateCallback,
		(XtCallbackProc) activateCB_menuBar_p1_b2,
		(XtPointer) NULL );



	/* Creation of menuBar_p1_b4 */
	menuBar_p1_b4 = XtVaCreateManagedWidget( "menuBar_p1_b4",
			xmPushButtonWidgetClass,
			menuBar_p1,
			RES_CONVERT( XmNlabelString, "Save as..." ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( menuBar_p1_b4, XmNactivateCallback,
		(XtCallbackProc) activateCB_menuBar_p1_b4,
		(XtPointer) NULL );



	/* Creation of menuBar_p1_b5 */
	menuBar_p1_b5 = XtVaCreateManagedWidget( "menuBar_p1_b5",
			xmSeparatorWidgetClass,
			menuBar_p1,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of menuBar_p1_b6 */
	menuBar_p1_b6 = XtVaCreateManagedWidget( "menuBar_p1_b6",
			xmPushButtonWidgetClass,
			menuBar_p1,
			RES_CONVERT( XmNlabelString, "Close" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( menuBar_p1_b6, XmNactivateCallback,
		(XtCallbackProc) activateCB_menuBar_p1_b6,
		(XtPointer) NULL );



	/* Creation of menuBar_top_b2 */
	menuBar_top_b2 = XtVaCreateManagedWidget( "menuBar_top_b2",
			xmCascadeButtonWidgetClass,
			menuBar1,
			RES_CONVERT( XmNlabelString, "File" ),
			XmNsubMenuId, menuBar_p1,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );


	/* Creation of menuBar1_p2 */
	menuBar1_p2_shell = XtVaCreatePopupShell ("menuBar1_p2_shell",
			xmMenuShellWidgetClass, menuBar1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	menuBar1_p2 = XtVaCreateWidget( "menuBar1_p2",
			xmRowColumnWidgetClass,
			menuBar1_p2_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of menuBar1_p2_b1 */
	menuBar1_p2_b1 = XtVaCreateManagedWidget( "menuBar1_p2_b1",
			xmPushButtonWidgetClass,
			menuBar1_p2,
			RES_CONVERT( XmNlabelString, "Clear" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( menuBar1_p2_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_menuBar1_p2_b1,
		(XtPointer) NULL );



	/* Creation of menuBar1_top_b3 */
	menuBar1_top_b3 = XtVaCreateManagedWidget( "menuBar1_top_b3",
			xmCascadeButtonWidgetClass,
			menuBar1,
			RES_CONVERT( XmNlabelString, "Edit" ),
			XmNsubMenuId, menuBar1_p2,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );


	/* Creation of menuBar1_p3 */
	menuBar1_p3_shell = XtVaCreatePopupShell ("menuBar1_p3_shell",
			xmMenuShellWidgetClass, menuBar1,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	menuBar1_p3 = XtVaCreateWidget( "menuBar1_p3",
			xmRowColumnWidgetClass,
			menuBar1_p3_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of menuBar1_p3_b1 */
	menuBar1_p3_b1 = XtVaCreateManagedWidget( "menuBar1_p3_b1",
			xmPushButtonWidgetClass,
			menuBar1_p3,
			RES_CONVERT( XmNlabelString, "Ferret Commands..." ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( menuBar1_p3_b1, XmNactivateCallback,
		(XtCallbackProc) activateCB_menuBar1_p3_b1,
		(XtPointer) NULL );



	/* Creation of menuBar1_top_b5 */
	menuBar1_top_b5 = XtVaCreateManagedWidget( "menuBar1_top_b5",
			xmCascadeButtonWidgetClass,
			menuBar1,
			RES_CONVERT( XmNlabelString, "Help" ),
			XmNsubMenuId, menuBar1_p3,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );


	/* Creation of scrolledWindowText1 */
	scrolledWindowText1 = XtVaCreateManagedWidget( "scrolledWindowText1",
			xmScrolledWindowWidgetClass,
			form12,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNx, 9,
			XmNy, 33,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNbottomOffset, 60,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNtopOffset, 35,
			XmNtopAttachment, XmATTACH_FORM,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of scrolledText1 */
	scrolledText1 = XtVaCreateManagedWidget( "scrolledText1",
			xmTextWidgetClass,
			scrolledWindowText1,
			XmNwidth, 474,
			XmNheight, 127,
			XmNeditMode, XmMULTI_LINE_EDIT ,
			RES_CONVERT( XmNbackground, "gray75" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNfontList, UxConvertFontList( "*times-medium-r-normal--*-120-*" ),
			NULL );
	XtAddCallback( scrolledText1, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scrolledText1,
		(XtPointer) NULL );
	XtAddCallback( scrolledText1, XmNmodifyVerifyCallback,
		(XtCallbackProc) modifyVerifyCB_scrolledText1,
		(XtPointer) NULL );
	XtAddCallback( scrolledText1, XmNmotionVerifyCallback,
		(XtCallbackProc) motionVerifyCB_scrolledText1,
		(XtPointer) NULL );
	XtAddCallback( scrolledText1, XmNgainPrimaryCallback,
		(XtCallbackProc) gainPrimaryCB_scrolledText1,
		(XtPointer) NULL );
	XtAddCallback( scrolledText1, XmNlosePrimaryCallback,
		(XtCallbackProc) losePrimaryCB_scrolledText1,
		(XtPointer) NULL );



	/* Creation of rowColumn18 */
	rowColumn18 = XtVaCreateManagedWidget( "rowColumn18",
			xmRowColumnWidgetClass,
			form12,
			XmNwidth, 141,
			XmNheight, 33,
			XmNx, 176,
			XmNy, 253,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNmarginHeight, 0,
			XmNbottomOffset, 22,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNleftPosition, 40,
			XmNleftOffset, -1,
			XmNleftAttachment, XmATTACH_POSITION,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of toggleButton68 */
	toggleButton68 = XtVaCreateManagedWidget( "toggleButton68",
			xmToggleButtonWidgetClass,
			rowColumn18,
			XmNx, 11,
			XmNy, 7,
			XmNwidth, 73,
			XmNheight, 17,
			XmNindicatorOn, FALSE,
			RES_CONVERT( XmNlabelString, "  " ),
			XmNlabelType, XmPIXMAP,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( toggleButton68, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton68,
		(XtPointer) NULL );



	/* Creation of toggleButton69 */
	toggleButton69 = XtVaCreateManagedWidget( "toggleButton69",
			xmToggleButtonWidgetClass,
			rowColumn18,
			XmNx, 31,
			XmNy, 8,
			XmNwidth, 93,
			XmNheight, 15,
			XmNindicatorOn, FALSE,
			RES_CONVERT( XmNlabelString, "  " ),
			XmNlabelType, XmPIXMAP,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( toggleButton69, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton69,
		(XtPointer) NULL );



	/* Creation of toggleButton70 */
	toggleButton70 = XtVaCreateManagedWidget( "toggleButton70",
			xmToggleButtonWidgetClass,
			rowColumn18,
			XmNx, 21,
			XmNy, 9,
			XmNwidth, 107,
			XmNheight, 26,
			XmNindicatorOn, FALSE,
			RES_CONVERT( XmNlabelString, "  " ),
			XmNlabelType, XmPIXMAP,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( toggleButton70, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton70,
		(XtPointer) NULL );



	/* Creation of label60 */
	label60 = XtVaCreateManagedWidget( "label60",
			xmLabelWidgetClass,
			form12,
			XmNx, 178,
			XmNy, 291,
			XmNwidth, 56,
			XmNheight, 13,
			RES_CONVERT( XmNlabelString, "Record" ),
			XmNtopOffset, 5,
			XmNtopWidget, rowColumn18,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftPosition, 39,
			XmNleftOffset, 1,
			XmNleftAttachment, XmATTACH_POSITION,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNfontList, UxConvertFontList( "-adobe-courier-bold-r-normal--12-120-75-75-m-70-iso8859-1" ),
			NULL );


	/* Creation of label61 */
	label61 = XtVaCreateManagedWidget( "label61",
			xmLabelWidgetClass,
			form12,
			XmNx, 230,
			XmNy, 291,
			XmNwidth, 46,
			XmNheight, 13,
			RES_CONVERT( XmNlabelString, "Stop" ),
			XmNtopOffset, 5,
			XmNtopWidget, rowColumn18,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftOffset, -4,
			XmNleftWidget, label60,
			XmNleftAttachment, XmATTACH_WIDGET,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNfontList, UxConvertFontList( "-adobe-courier-bold-r-normal--12-120-75-75-m-70-iso8859-1" ),
			NULL );


	/* Creation of label62 */
	label62 = XtVaCreateManagedWidget( "label62",
			xmLabelWidgetClass,
			form12,
			XmNx, 278,
			XmNy, 291,
			XmNwidth, 37,
			XmNheight, 13,
			RES_CONVERT( XmNlabelString, "Run" ),
			XmNtopOffset, 5,
			XmNtopWidget, rowColumn18,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftOffset, 2,
			XmNleftWidget, label61,
			XmNleftAttachment, XmATTACH_WIDGET,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNfontList, UxConvertFontList( "-adobe-courier-bold-r-normal--12-120-75-75-m-70-iso8859-1" ),
			NULL );

	XtVaSetValues(menuBar1,
			XmNmenuHelpWidget, menuBar1_top_b5,
			NULL );

	XtVaSetValues(menuBar1_p3,
			XmNmenuHelpWidget, menuBar1_p3_b1,
			NULL );



	return ( MacroManager );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_MacroManager( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedMacroManager == NULL) {
		rtrn = _Uxbuild_MacroManager();

		InitPixmaps();
			gSavedMacroManager = rtrn;
		}
		else
			rtrn = gSavedMacroManager;
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

