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
	ErrorLog.c

       Associated Header file: ErrorLog.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/Text.h>
#include <Xm/ScrolledW.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "ErrorLog_code.h"


static	Widget	form1;
static	Widget	pushButton1;
static	Widget	pushButton2;
static	Widget	pushButton3;
static	Widget	scrolledWindowText1;
static	Widget	scrolledText1;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "ErrorLog.h"
#undef CONTEXT_MACRO_ACCESS

Widget	ErrorLog;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#include "ErrorLog_code.c"

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	valueChangedCB_scrolledText1(
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

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_ErrorLog()
{
	Widget		_UxParent;


	/* Creation of ErrorLog */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	ErrorLog = XtVaCreatePopupShell( "ErrorLog",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 755,
			XmNheight, 279,
			XmNx, 138,
			XmNy, 358,
			XmNiconName, "Ferret: Error Log",
			XmNtitle, "Error Log",
			XmNallowShellResize, TRUE,
			XmNdeleteResponse, XmUNMAP,
			NULL );


	/* Creation of form1 */
	form1 = XtVaCreateManagedWidget( "form1",
			xmFormWidgetClass,
			ErrorLog,
			XmNwidth, 755,
			XmNheight, 279,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 0,
			XmNy, 0,
			XmNunitType, XmPIXELS,
			NULL );


	/* Creation of pushButton1 */
	pushButton1 = XtVaCreateManagedWidget( "pushButton1",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Save" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_NONE,
			XmNheight, 30,
			XmNwidth, 100,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			NULL );
	XtAddCallback( pushButton1, XmNactivateCallback,
		(XtCallbackProc) JC_SaveButton_CB,
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
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, -50,
			XmNleftPosition, 50,
			NULL );
	XtAddCallback( pushButton2, XmNactivateCallback,
		(XtCallbackProc) JC_ClearButton_CB,
		(XtPointer) NULL );



	/* Creation of pushButton3 */
	pushButton3 = XtVaCreateManagedWidget( "pushButton3",
			xmPushButtonWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Dismiss" ),
			XmNheight, 30,
			XmNwidth, 100,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNleftAttachment, XmATTACH_NONE,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNbottomOffset, 10,
			NULL );
	XtAddCallback( pushButton3, XmNactivateCallback,
		(XtCallbackProc) JC_ErrorLog_Popdown,
		(XtPointer) NULL );



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
			XmNbottomOffset, 10,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, pushButton1,
			XmNtopAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
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




	return ( ErrorLog );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_ErrorLog( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedErrLog == NULL) {
		rtrn = _Uxbuild_ErrorLog();

		gSavedErrLog = rtrn;
		} else
			rtrn = gSavedErrLog;
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

