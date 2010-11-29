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
 * OpenFile_code.c
 *
 * John Osborne,
 * Jonathan Callahan
 * Apr 4th 1996
 *
 * This file contains the auxiliary functions which are included by
 * OpenFile.c.
 *
 * 96.12.12 Removed JC_II_NewDataset in OpenOK().  All synching happens in ferret_command.
 */

/* .................... Function Definitions .................... */


static void ActivateCB(wid, clientData, cbArg)
Widget wid;
XtPointer clientData, cbArg;
{
	char *valText = XmTextFieldGetString(wid);

	if (strlen(valText))
		strcpy(DSText, valText);
	else {
		XBell(XtDisplay(UxGetWidget(FerretMainWd)), 50);
		XmTextSetString(wid, DSText);
	}
}


static void InitialList()
{
     XmString motif_string;
     int i=0;

     list_mvfront(GLOBAL_DatasetNameList);

     for ( i=0; i<list_size(GLOBAL_DatasetNameList); i++, list_mvnext(GLOBAL_DatasetNameList) ) {
	  motif_string = XmStringCreateSimple(list_curr(GLOBAL_DatasetNameList));
	  XmListAddItem(UxGetWidget(scrolledList1), motif_string, 0);
     }

     XmListSelectPos(UxGetWidget(scrolledList1), 1, True);

     MaintainBtns();
}

static void MaintainBtns()
{
	if (strlen(DSText))
		XtSetSensitive(pushButton2, True);
	else
		XtSetSensitive(pushButton2, False);
}

static void ListBrowserCB(Widget wid, XtPointer client_data, XtPointer call_data)
{
	char *tempText;
	XmListCallbackStruct *cbs = (XmListCallbackStruct *) call_data;
	
	/* get text selection from list */
	XmStringGetLtoR(cbs->item, XmSTRING_DEFAULT_CHARSET, &tempText);

	/* construct DSText */
	strcpy(DSText, tempText);

	/* put this into edit text field */
	XmTextFieldSetString(UxGetWidget(textField1), DSText);
	
	MaintainBtns();
	XtFree(tempText); /* allocated with XmStringGetLtoR() */
}

static void CancelOpen()
{
  	XtDestroyWidget(UxGetWidget(OpenFile));
}

static void OpenOK()
{
  /* 	upped cmd from 80 to 256 chars - was causing crashes w/ dods datasets
        *kob* 3/25/99  and use macro MAX_NAME_LENGTH */

	char cmd[MAX_NAME_LENGTH];

	if ( JC_String_EndsWithTag(DSText, ".cdf") || JC_String_EndsWithTag(DSText, ".nc"))
	  sprintf(cmd, "USE %s", DSText);
	else 
	  sprintf(cmd, "SET DATA %s", DSText);

	ferret_command(cmd, IGNORE_COMMAND_WIDGET);

  	XtDestroyWidget(UxGetWidget(OpenFile));
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/
/*
static	void	destroyCB_OpenFile(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedOpenFile = NULL;
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
	OpenOK();
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
	XtDestroyWidget(UxGetWidget(OpenFile));
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
	Open_Save_dset = create_Open_Save_dset(NO_PARENT);
	XtPopup(UxGetWidget(Open_Save_dset), no_grab);
	
	
	}
}

static	void	activateCB_textField1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ActivateCB(UxWidget, UxClientData, UxCallbackArg);
	OpenOK();
	
	}
}

static	void	losingFocusCB_textField1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ActivateCB(UxWidget, UxClientData, UxCallbackArg);
	
	}
}

static	void	singleSelectionCB_scrolledList1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ListBrowserCB(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	defaultActionCB_scrolledList1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ListBrowserCB(UxWidget, UxClientData, UxCallbackArg);
	OpenOK();
	}
}

*/
