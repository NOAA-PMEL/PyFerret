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



/* 
 * ErrorLog_code.c
 *
 * John Osborne,
 * Jonathan Callahan
 * Dec 16th 1996
 *
 * This file contains the auxiliary functions which are included by
 * ErrorLog.c.
 *
 */
 
/* .................... Function Definitions .................... */
 
 
void ferret_error(smPtr iBuffer)
{
  int i, result;
  int numErrs;
  char errText[512];

  /* got a message from ferret, display it in an alert */
  numErrs = iBuffer->numStrings;

  strcpy(errText, "");
  strcpy(errText, sBuffer->text);
  result = PresentWarnDialog(UxGetWidget(FerretMainWd), errText);
  ferret_logerr(errText);
}


void ferret_logerr(char *text)
{
  if (!gSavedErrLog) {
    /* no output window available--create a new one */
    ErrorLog = create_ErrorLog(NO_PARENT);
    epos = 0;
  }
  strcat(text, "\n");
  XmTextInsert(UxGetWidget(scrolledText1), epos, text);
  epos = epos + strlen(text);
}


void ferret_pause(void)
{
  int result;
  char errText[512];

  strcpy(errText, "Script Paused--Click OK to Continue");
  result = PresentWarnDialog(UxGetWidget(FerretMainWd), errText);
}


static int PresentWarnDialog(Widget parent, char *itext)
{
  int n = 0, i;
  static Widget dialog; /* static to avoid multiple creation */
  XmString text;
  static int answer;
  Display *dis;
  XtAppContext app;
  Arg args[5];

  answer = 0;
  for (i=0; i<strlen(itext);i++) {
    if (itext[i] == '\n' || itext[i] == '\r')
      itext[i] = ' ';
  }

  text = XmStringCreateLocalized (itext);

  if (!dialog) {
    /* customize the dialog */
    XtSetArg(args[n], XmNautoUnmanage, False); n++;

    dialog = (Widget)XmCreateQuestionDialog (parent, "Ferret Error", args, n);
    XtVaSetValues (dialog,
		   XmNdialogType, XmDIALOG_ERROR,
		   NULL);
    XtVaSetValues (dialog,
		   XmNtitle, "Ferret Error",
		   NULL);
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

  XtVaSetValues (dialog,
		 XmNmessageString, text,
		 NULL);
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
static void response(Widget widget, XtPointer client_data, XtPointer call_data)
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
  ;
}


static void JC_SaveButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
  Open_Save_err = create_Open_Save_err(NO_PARENT);
  SaveAsErrLogFile();
}


static void JC_ClearButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
  XtVaSetValues(UxGetWidget(scrolledText1),
		XmNvalue, "",
		NULL);
}


void JC_ErrorLog_Popdown( Widget wid, XtPointer client_data, XtPointer call_data )
{
  XtPopdown(UxGetWidget(ErrorLog));
}


void ErrLogSaveAsOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg)
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
    result = AskUser2(UxGetWidget(ErrorLog), question, yes, no, defaultBtn);
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
  XtDestroyWidget(UxGetWidget(Open_Save_err));

  XtFree(pathName); /* allocated with XtMalloc() */
  XtFree(contents); /* allocated with XmTextGetString() */
}


void ErrLogCancel(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg)
{
  /* pop down the interface */
  XtDestroyWidget(UxGetWidget(Open_Save_err));
}


void SaveAsErrLogFile()
{
  XmString mask, selection, dir;

  XtVaGetValues(UxGetWidget(fileSelectionBox8),
		XmNdirectory, &dir,
		NULL);

  mask = XmStringCreateLocalized("*.err");
  XtVaSetValues(UxGetWidget(fileSelectionBox8),
		XmNdirMask, mask,
		NULL);

  selection = XmStringConcat(dir, XmStringCreateLocalized(".err"));
  XtVaSetValues(UxGetWidget(fileSelectionBox8),
		XmNtextString, selection,
		NULL);

  /* popup Open file */
  XtPopup(UxGetWidget(Open_Save_err), XtGrabNone);

  XmStringFree(mask);
  XmStringFree(selection);
  XmStringFree(dir);
}
