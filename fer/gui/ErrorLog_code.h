/* 
 * ErrorLog_code.h
 *
 * John Osborne
 * Jonathan Callahan
 * December 16th 1996
 *
 * This file contains the necessary header information which is included by
 * ErrorLog.h.
 *
 */

/*.....     includes     .....*/
#include <sys/types.h>
#include <sys/stat.h>
#include <Xm/PushB.h>

#include "ferret_structures.h"
#include "ferret_shared_buffer.h"


/*.....     defines     .....*/
#define YES 1
#define NO 2
#define CANCEL 3

/*.....     variables     .....*/
static XmTextPosition epos=0;

swidget ErrorLog, gSavedErrLog=NULL;

extern swidget FerretMainWd, fileSelectionBox8, Open_Save_err;

/*.....     functions     .....*/
static void InitialState(void);
static int PresentWarnDialog(Widget parent, char *itext);
static void response(Widget widget, XtPointer client_data, XtPointer call_data);
static void JC_ClearButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_SaveButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_ErrorLog_Popdown( Widget wid, XtPointer client_data, XtPointer call_data );

swidget create_ErrorLog(swidget UxParent);
void ErrLogSaveAsOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
void ErrLogCancel(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
void SaveAsErrLogFile(void);
void ferret_error(smPtr iBuffer);
void ferret_logerr(char *text);
void ferret_pause(void);

extern char *CollectToReturn(char *targetStr, char *subStr);
extern swidget create_Open_Save_err(swidget UxParent);


