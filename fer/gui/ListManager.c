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
	ListManager.c

       Associated Header file: ListManager.h
*******************************************************************************/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <Xm/PushB.h>
#include "ferret_structures.h"

#define YES 1
#define NO 2
#define CANCEL 3
#define REPL_OUT_TEXT 0
#define APND_OUT_TEXT 1
#define NEW_OUT_TEXT 3

/* globals */
swidget ListManager, gSavedListManager=NULL;
extern swidget FerretMainWd, fileSelectionBox6;
swidget gAllListManagers[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
NULL};
extern swidget Open_Save_list;
swidget gCurrTextWidget = NULL;
swidget gTextWidgetPool[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
NULL};
int gCurrList=0, gNumOutputWindows=0;
static XmTextPosition pos=0; /*[10]={0,0,0,0,0,0,0,0,0,0}*/
int gOlineCount = 0;

/* prototypes */
swidget create_ListManager(swidget UxParent);
extern swidget create_Open_Save_list(swidget UxParent);
static void InitialState(void);
static void ClearRadioGroup1(void);
void ListSaveAsOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
void ListCancel(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
void SaveAsListFile(void);
extern void ferret_command(char *cmdText, int cmdMode);
extern char *CollectToReturn(char *targetStr, char *subStr);
extern int AskUser(Widget parent, char *question, char *ans1, char *ans2, int default_ans);
static void response(Widget widget, XtPointer client_data, XtPointer call_data);
void AddWIDToPool(swidget wid);
void DeleteWIDFromPool(swidget wid);
swidget GetWIDFromPool(void);
void SetWID(swidget wid);
void ShowCursorPos(void);
static int PresentWarnDialog(Widget parent, char *itext, int mode);
static void response(Widget widget, XtPointer client_data, XtPointer call_data);


static	Widget	form1;
static	Widget	scrolledWindowText1;
static	Widget	scrolledText1;
static	Widget	pushButton1;
static	Widget	pushButton2;
static	Widget	pushButton3;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "ListManager.h"
#undef CONTEXT_MACRO_ACCESS

Widget	ListManager;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static int PresentWarnDialog(parent, itext, mode)
Widget parent;
char *itext;
int mode;
{
    int n = 0;
    static Widget dialog; /* static to avoid multiple creation */
    XmString text;
    static int answer;
    Display *dis;
    XtAppContext app;
    Arg args[5];

    answer = 0;
    text = XmStringCreateLocalized (itext);

    if (!dialog) {
	/* customize the dialog */
	XtSetArg(args[n], XmNautoUnmanage, False); n++;
	XtSetArg(args[n], XmNmessageString, text); n++;

        dialog = (Widget)XmCreateQuestionDialog (parent, "Ferret Alert", args, n);
	switch (mode) {
		case FWARN_ERROR:
        		XtVaSetValues (dialog,
           			 XmNdialogType, XmDIALOG_ERROR,
           			 NULL);
			break;
		case FWARN_INFO:
        		XtVaSetValues (dialog,
           			 XmNdialogType, XmDIALOG_ERROR,
           			 NULL);
			break;
		case FWARN_WARNING:
        		XtVaSetValues (dialog,
           			 XmNdialogType, XmDIALOG_ERROR,
           			 NULL);
			break;
		case FWARN_MSG:
        		XtVaSetValues (dialog,
           			 XmNdialogType, XmDIALOG_ERROR,
           			 NULL);
			break;
		default:
        		XtVaSetValues (dialog,
           			 XmNdialogType, XmDIALOG_ERROR,
           			 NULL);

	}
        XtVaSetValues (dialog,
            XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL,
            NULL);
        XtUnmanageChild((Widget)XmMessageBoxGetChild (dialog, XmDIALOG_HELP_BUTTON));

	XtUnmanageChild((Widget)XmMessageBoxGetChild (dialog, XmDIALOG_CANCEL_BUTTON));
   	XtVaSetValues (dialog,
       	 		XmNdefaultButtonType,  XmDIALOG_OK_BUTTON,
        		NULL);

        XtAddCallback (dialog, XmNokCallback, response, &answer);
    }

    dis = XtDisplayOfObject(dialog);
    app = XtDisplayToApplicationContext(dis);

    XmStringFree (text);
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
static void response(widget, client_data, call_data)
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

#ifdef NO_ENTRY_NAME_UNDERSCORES
void ferret_warn_in_window(text, mode)
#else
void ferret_warn_in_window_(text, mode)
#endif
char *text;
int mode;
{
	int result;

	/* got a message from ferret, display it in an alert */
	result = PresentWarnDialog(UxGetWidget(FerretMainWd), text, mode);
}

#ifdef NO_ENTRY_NAME_UNDERSCORES
void ferret_list_in_window(text, mode)
#else
void ferret_list_in_window_(text, mode)
#endif
char *text;
int mode;
{
	if (!gSavedListManager) {
		/* no output window available--create a new one */
		ListManager = create_ListManager(NO_PARENT);
		mode = APND_OUT_TEXT;
		pos = 0;
	}
	strcat(text, "\n");

	XmTextInsert(UxGetWidget(scrolledText1), pos, text);
	pos = pos + strlen(text);
	
	if (gOlineCount % 100 == 0) {
		XtPopup(UxGetWidget(ListManager), no_grab);
		ShowCursorPos();
		gOlineCount = 0;
	}
	else
		gOlineCount++;
}


void ShowCursorPos()
{
	XtVaSetValues(UxGetWidget(scrolledText1),
		XmNcursorPosition, pos,
		NULL);
	XmTextShowPosition(UxGetWidget(scrolledText1), pos);
}

static void InitialState()
{
	;
}

extern void ListSaveAsOK(UxWidget, UxClientData, UxCallbackArg)
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
		result = AskUser2(UxGetWidget(ListManager), question, yes, no, defaultBtn);
		if (result == NO)
			return;	
	}

	/* open the file */
	outFile = fopen(pathName, "w");
	
	/* get a pointer to contour text */
	contents = XmTextGetString(UxGetWidget(scrolledText1));

	/* write text to file */
	io = fwrite(contents, sizeof(char), strlen(contents), outFile);

	/* close file */
	io = fclose(outFile);

	/* pop down the interface */
	XtDestroyWidget(UxGetWidget(Open_Save_list));

	XtFree(pathName); /* allocated with XtMalloc() */
	XtFree(contents); /* allocated with XmTextGetString() */
}

void ListCancel(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	/* pop down the interface */
	XtDestroyWidget(UxGetWidget(Open_Save_list));
}

void SaveAsListFile()
{
	XmString mask, selection, dir;

	XtVaGetValues(UxGetWidget(fileSelectionBox6),
		XmNdirectory, &dir,
		NULL);

	mask = XmStringCreateLocalized("*.lst");
	XtVaSetValues(UxGetWidget(fileSelectionBox6),
		XmNdirMask, mask,
		NULL);

	selection = XmStringConcat(dir, XmStringCreateLocalized(".lst"));
	XtVaSetValues(UxGetWidget(fileSelectionBox6),
		XmNtextString, selection,
		NULL);

	/* popup Open file */
	XtPopup(UxGetWidget(Open_Save_list), XtGrabNone); 
	XmStringFree(mask); 
	XmStringFree(selection);
	XmStringFree(dir);
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	popupCB_ListManager(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	InitialState();
}

static	void	destroyCB_ListManager(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedListManager = NULL;
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

static	void	activateCB_pushButton1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	  Open_Save_list = create_Open_Save_list(NO_PARENT);
	  SaveAsListFile();
	
	}
}

static	void	activateCB_pushButton2(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	  XtVaSetValues(UxGetWidget(scrolledText1),
	        XmNvalue, "",
	        NULL);
	
	}
}

static	void	activateCB_pushButton3(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XtDestroyWidget(UxGetWidget(ListManager));
	
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_ListManager()
{
	Widget		_UxParent;


	/* Creation of ListManager */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	ListManager = XtVaCreatePopupShell( "ListManager",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 876,
			XmNheight, 293,
			XmNx, 128,
			XmNy, 110,
			XmNiconName, "Ferret Output",
			XmNtitle, "Ferret Output",
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( ListManager, XmNpopupCallback,
		(XtCallbackProc) popupCB_ListManager,
		(XtPointer) NULL );
	XtAddCallback( ListManager, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_ListManager,
		(XtPointer) NULL );



	/* Creation of form1 */
	form1 = XtVaCreateManagedWidget( "form1",
			xmFormWidgetClass,
			ListManager,
			XmNwidth, 876,
			XmNheight, 293,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 33,
			XmNy, 30,
			XmNunitType, XmPIXELS,
			NULL );


	/* Creation of scrolledWindowText1 */
	scrolledWindowText1 = XtVaCreateManagedWidget( "scrolledWindowText1",
			xmScrolledWindowWidgetClass,
			form1,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNx, 8,
			XmNy, 39,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNbottomOffset, 50,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of scrolledText1 */
	scrolledText1 = XtVaCreateManagedWidget( "scrolledText1",
			xmTextWidgetClass,
			scrolledWindowText1,
			XmNwidth, 474,
			XmNheight, 127,
			XmNeditMode, XmMULTI_LINE_EDIT ,
			XmNautoShowCursorPosition, FALSE,
			XmNeditable, FALSE,
			XmNcolumns, 80,
			NULL );
	XtAddCallback( scrolledText1, XmNmodifyVerifyCallback,
		(XtCallbackProc) modifyVerifyCB_scrolledText1,
		(XtPointer) NULL );



	/* Creation of pushButton1 */
	pushButton1 = XtVaCreateManagedWidget( "pushButton1",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Save" ),
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_NONE,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNheight, 30,
			XmNwidth, 100,
			XmNleftOffset, 10,
			NULL );
	XtAddCallback( pushButton1, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton1,
		(XtPointer) NULL );



	/* Creation of pushButton2 */
	pushButton2 = XtVaCreateManagedWidget( "pushButton2",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Clear" ),
			XmNheight, 30,
			XmNwidth, 100,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNleftOffset, -50,
			XmNleftPosition, 50,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNrightAttachment, XmATTACH_NONE,
			XmNtopAttachment, XmATTACH_NONE,
			NULL );
	XtAddCallback( pushButton2, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton2,
		(XtPointer) NULL );



	/* Creation of pushButton3 */
	pushButton3 = XtVaCreateManagedWidget( "pushButton3",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Dismiss" ),
			XmNheight, 30,
			XmNwidth, 100,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			NULL );
	XtAddCallback( pushButton3, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton3,
		(XtPointer) NULL );




	return ( ListManager );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_ListManager( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedListManager == NULL) {
		rtrn = _Uxbuild_ListManager();

		gSavedListManager = rtrn;
		}
		else
			rtrn = gSavedListManager;
		
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

