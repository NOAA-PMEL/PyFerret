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
	OpenFile.c

       Associated Header file: OpenFile.h
*******************************************************************************/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/List.h>
#include <Xm/ScrolledW.h>
#include <Xm/TextF.h>
#include <Xm/PushB.h>
#include <Xm/Label.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "OpenFile_code.h"
/* added below include to get MAX_NAME_LENGTH macro 3/99 *kob* */
#include "ferret_structures.h"

static	Widget	form1;
static	Widget	label1;
static	Widget	pushButton1;
static	Widget	pushButton3;
static	Widget	pushButton2;
static	Widget	textField1;
static	Widget	scrolledWindowList1;
static	Widget	scrolledList1;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "OpenFile.h"
#undef CONTEXT_MACRO_ACCESS

Widget	OpenFile;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#include "OpenFile_code.c"

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

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

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_OpenFile()
{
	Widget		_UxParent;


	/* Creation of OpenFile */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	OpenFile = XtVaCreatePopupShell( "OpenFile",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 460,
			XmNheight, 253,
			XmNx, 268,
			XmNy, 317,
			XmNiconName, "Ferret: Open Dataset",
			XmNtitle, "Ferret: Open Dataset",
			XmNminHeight, 200,
			XmNminWidth, 340,
			NULL );
	XtAddCallback( OpenFile, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_OpenFile,
		(XtPointer) NULL );



	/* Creation of form1 */
	form1 = XtVaCreateManagedWidget( "form1",
			xmFormWidgetClass,
			OpenFile,
			XmNresizePolicy, XmRESIZE_ANY,
			XmNunitType, XmPIXELS,
			XmNwidth, 500,
			XmNheight, 253,
			NULL );


	/* Creation of label1 */
	label1 = XtVaCreateManagedWidget( "label1",
			xmLabelWidgetClass,
			form1,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Datasets:" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 5,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of pushButton1 */
	pushButton1 = XtVaCreateManagedWidget( "pushButton1",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Apply" ),
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNheight, 30,
			XmNwidth, 100,
			XmNleftOffset, 10,
			XmNx, 10,
			NULL );
	XtAddCallback( pushButton1, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton1,
		(XtPointer) NULL );



	/* Creation of pushButton3 */
	pushButton3 = XtVaCreateManagedWidget( "pushButton3",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Dismiss" ),
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNresizable, FALSE,
			XmNtopAttachment, XmATTACH_NONE,
			XmNleftAttachment, XmATTACH_NONE,
			XmNheight, 30,
			XmNwidth, 100,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			NULL );
	XtAddCallback( pushButton3, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton3,
		(XtPointer) NULL );



	/* Creation of pushButton2 */
	pushButton2 = XtVaCreateManagedWidget( "pushButton2",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Search" ),
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNresizable, FALSE,
			XmNheight, 30,
			XmNwidth, 100,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, -50,
			XmNleftPosition, 50,
			NULL );
	XtAddCallback( pushButton2, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton2,
		(XtPointer) NULL );



	/* Creation of textField1 */
	textField1 = XtVaCreateManagedWidget( "textField1",
			xmTextFieldWidgetClass,
			form1,
			XmNsensitive, TRUE,
			XmNleftOffset, 5,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_NONE,
			XmNx, 10,
			XmNwidth, 285,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomOffset, 10,
			XmNbottomWidget, pushButton1,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			NULL );
	XtAddCallback( textField1, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField1,
		(XtPointer) NULL );
	XtAddCallback( textField1, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField1,
		(XtPointer) NULL );



	/* Creation of scrolledWindowList1 */
	scrolledWindowList1 = XtVaCreateManagedWidget( "scrolledWindowList1",
			xmScrolledWindowWidgetClass,
			form1,
			XmNleftOffset, 7,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 7,
			XmNtopWidget, label1,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNheight, 136,
			XmNx, 10,
			XmNwidth, 282,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomOffset, 5,
			XmNbottomWidget, textField1,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 7,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			NULL );


	/* Creation of scrolledList1 */
	scrolledList1 = XtVaCreateManagedWidget( "scrolledList1",
			xmListWidgetClass,
			scrolledWindowList1,
			XmNheight, 136,
			XmNselectionPolicy, XmSINGLE_SELECT,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			NULL );
	XtAddCallback( scrolledList1, XmNsingleSelectionCallback,
		(XtCallbackProc) singleSelectionCB_scrolledList1,
		(XtPointer) NULL );
	XtAddCallback( scrolledList1, XmNdefaultActionCallback,
		(XtCallbackProc) defaultActionCB_scrolledList1,
		(XtPointer) NULL );




	return ( OpenFile );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_OpenFile( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedOpenFile == NULL) {
		rtrn = _Uxbuild_OpenFile();

		InitialList();
		}
		else
			rtrn = gSavedOpenFile;
		
		/* set the hi rez size */
		if (gHiRez && !gSavedOpenFile) {
			Dimension width, height;
				
			XtVaGetValues(UxGetWidget(form1),
				XmNwidth, &width,
				XmNheight, &height,
				NULL);
			width = 1.27 * width;
			height = 1.1 * height;
				
			XtVaSetValues(UxGetWidget(form1),
				XmNwidth, width,
				XmNheight, height,
				NULL);
		}
		if (!gSavedOpenFile)
			gSavedOpenFile = rtrn;
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

