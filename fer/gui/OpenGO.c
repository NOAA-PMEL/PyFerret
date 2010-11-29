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
	OpenGO.c

       Associated Header file: OpenGO.h
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
#include <Xm/RowColumn.h>
#include <Xm/Label.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "OpenGO_code.h"


static	Widget	form17;
static	Widget	label49;
static	Widget	rowColumn4;
static	Widget	pushButton24;
static	Widget	pushButton25;
static	Widget	pushButton19;
static	Widget	textField14;
static	Widget	scrolledWindowList3;
static	Widget	scrolledList3;
static	Widget	label13;
static	Widget	rowColumn14;
static	Widget	pushButton23;
static	Widget	label53;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "OpenGO.h"
#undef CONTEXT_MACRO_ACCESS

Widget	OpenGO;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#include "OpenGO_code.c"

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	destroyCB_OpenGO(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedOpenGO = NULL;
}

static	void	activateCB_pushButton24(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	XtDestroyWidget(UxGetWidget(OpenGO));
	}
}

static	void	activateCB_pushButton25(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ViewCB();
	}
}

static	void	activateCB_pushButton19(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	GOOpenOK();
	}
}

static	void	activateCB_textField14(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ActivateCB(UxWidget, UxClientData, UxCallbackArg);
	GOOpenOK();
	}
}

static	void	losingFocusCB_textField14(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	ActivateCB(UxWidget, UxClientData, UxCallbackArg);
}

static	void	singleSelectionCB_scrolledList3(
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

static	void	defaultActionCB_scrolledList3(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ListBrowserCB(UxWidget, UxClientData, UxCallbackArg);
	GOOpenOK();
	}
}

static	void	activateCB_pushButton23(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	OpenGOFile = create_OpenGOFile(NO_PARENT);
	XtPopup(UxGetWidget(OpenGOFile), no_grab);
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_OpenGO()
{
	Widget		_UxParent;


	/* Creation of OpenGO */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	OpenGO = XtVaCreatePopupShell( "OpenGO",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 398,
			XmNheight, 249,
			XmNx, 24,
			XmNy, 497,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNiconName, "Ferret: Run Macro",
			XmNtitle, "Ferret Run Macro",
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( OpenGO, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_OpenGO,
		(XtPointer) NULL );



	/* Creation of form17 */
	form17 = XtVaCreateManagedWidget( "form17",
			xmFormWidgetClass,
			OpenGO,
			XmNresizePolicy, XmRESIZE_ANY,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNwidth, 398,
			XmNheight, 249,
			NULL );


	/* Creation of label49 */
	label49 = XtVaCreateManagedWidget( "label49",
			xmLabelWidgetClass,
			form17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Scripts:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 3,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of rowColumn4 */
	rowColumn4 = XtVaCreateManagedWidget( "rowColumn4",
			xmRowColumnWidgetClass,
			form17,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNorientation, XmHORIZONTAL,
			XmNpacking, XmPACK_COLUMN,
			XmNentryAlignment, XmALIGNMENT_CENTER,
			XmNmarginHeight, 10,
			XmNspacing, 10,
			XmNleftPosition, 10,
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNy, 202,
			XmNtopOffset, 202,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of pushButton24 */
	pushButton24 = XtVaCreateManagedWidget( "pushButton24",
			xmPushButtonWidgetClass,
			rowColumn4,
			RES_CONVERT( XmNlabelString, "Done" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNalignment, XmALIGNMENT_CENTER,
			XmNx, -10,
			XmNy, 11,
			NULL );
	XtAddCallback( pushButton24, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton24,
		(XtPointer) NULL );



	/* Creation of pushButton25 */
	pushButton25 = XtVaCreateManagedWidget( "pushButton25",
			xmPushButtonWidgetClass,
			rowColumn4,
			RES_CONVERT( XmNlabelString, "View" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNalignment, XmALIGNMENT_CENTER,
			XmNx, 27,
			XmNy, 11,
			NULL );
	XtAddCallback( pushButton25, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton25,
		(XtPointer) NULL );



	/* Creation of pushButton19 */
	pushButton19 = XtVaCreateManagedWidget( "pushButton19",
			xmPushButtonWidgetClass,
			rowColumn4,
			RES_CONVERT( XmNlabelString, "Run Script" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNalignment, XmALIGNMENT_CENTER,
			XmNx, 55,
			XmNy, 9,
			NULL );
	XtAddCallback( pushButton19, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton19,
		(XtPointer) NULL );



	/* Creation of textField14 */
	textField14 = XtVaCreateManagedWidget( "textField14",
			xmTextFieldWidgetClass,
			form17,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNleftAttachment, XmATTACH_FORM,
			XmNleftOffset, 30,
			XmNtopOffset, 168,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );
	XtAddCallback( textField14, XmNactivateCallback,
		(XtCallbackProc) activateCB_textField14,
		(XtPointer) NULL );
	XtAddCallback( textField14, XmNlosingFocusCallback,
		(XtCallbackProc) losingFocusCB_textField14,
		(XtPointer) NULL );



	/* Creation of scrolledWindowList3 */
	scrolledWindowList3 = XtVaCreateManagedWidget( "scrolledWindowList3",
			xmScrolledWindowWidgetClass,
			form17,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNshadowThickness, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 150,
			XmNrightAttachment, XmATTACH_FORM,
			XmNtopOffset, 2,
			XmNtopWidget, label49,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNheight, 130,
			NULL );


	/* Creation of scrolledList3 */
	scrolledList3 = XtVaCreateManagedWidget( "scrolledList3",
			xmListWidgetClass,
			scrolledWindowList3,
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNselectionPolicy, XmSINGLE_SELECT,
			XmNheight, 136,
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120*" ),
			NULL );
	XtAddCallback( scrolledList3, XmNsingleSelectionCallback,
		(XtCallbackProc) singleSelectionCB_scrolledList3,
		(XtPointer) NULL );
	XtAddCallback( scrolledList3, XmNdefaultActionCallback,
		(XtCallbackProc) defaultActionCB_scrolledList3,
		(XtPointer) NULL );



	/* Creation of label13 */
	label13 = XtVaCreateManagedWidget( "label13",
			xmLabelWidgetClass,
			form17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Other Scripts:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 12,
			XmNleftWidget, scrolledWindowList3,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 30,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of rowColumn14 */
	rowColumn14 = XtVaCreateManagedWidget( "rowColumn14",
			xmRowColumnWidgetClass,
			form17,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNentryAlignment, XmALIGNMENT_CENTER,
			XmNleftOffset, 21,
			XmNleftWidget, scrolledWindowList3,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 54,
			XmNtopAttachment, XmATTACH_FORM,
			XmNspacing, 40,
			NULL );


	/* Creation of pushButton23 */
	pushButton23 = XtVaCreateManagedWidget( "pushButton23",
			xmPushButtonWidgetClass,
			rowColumn14,
			RES_CONVERT( XmNlabelString, "Select..." ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNalignment, XmALIGNMENT_CENTER,
			NULL );
	XtAddCallback( pushButton23, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton23,
		(XtPointer) NULL );



	/* Creation of label53 */
	label53 = XtVaCreateManagedWidget( "label53",
			xmLabelWidgetClass,
			form17,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "GO" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 7,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 0,
			XmNrightWidget, NULL,
			XmNrightAttachment, XmATTACH_NONE,
			XmNtopAttachment, XmATTACH_FORM,
			XmNtopOffset, 177,
			NULL );



	return ( OpenGO );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_OpenGO( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedOpenGO == NULL) {
		rtrn = _Uxbuild_OpenGO();

		InitialList();
		}
		else
			rtrn = gSavedOpenGO;
		
		/* set the hi rez size */
		if (gHiRez && !gSavedOpenGO) {
			Dimension width, height;
				
			XtVaGetValues(UxGetWidget(form17),
				XmNwidth, &width,
				XmNheight, &height,
				NULL);
			width = 1.2 * width;
			height = 1.1 * height;
				
			XtVaSetValues(UxGetWidget(form17),
				XmNwidth, width,
				XmNheight, height,
				NULL);
		}
		if (!gSavedOpenGO)
			gSavedOpenGO = rtrn;
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

