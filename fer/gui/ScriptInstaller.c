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
	ScriptInstaller.c

       Associated Header file: ScriptInstaller.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/RowColumn.h>
#include <Xm/List.h>
#include <Xm/PushB.h>
#include <Xm/Label.h>
#include <Xm/Form.h>
#include <Xm/ScrolledW.h>
#include <X11/Shell.h>



static	Widget	scrolledWindowText7;
static	Widget	form31;
static	Widget	label34;
static	Widget	pushButton40;
static	Widget	scrolledWindowList2;
static	Widget	scrolledList2;
static	Widget	rowColumn40;
static	Widget	pushButton37;
static	Widget	pushButton39;
static	Widget	label52;
static	Widget	scrolledWindowList5;
static	Widget	scrolledList5;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "ScriptInstaller.h"
#undef CONTEXT_MACRO_ACCESS

Widget	ScriptInstaller;

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	activateCB_pushButton40(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget Viewports;
	
	XtPopdown(UxGetWidget(Viewports));
	}
}

static	void	singleSelectionCB_scrolledList2(
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

static	void	defaultActionCB_scrolledList2(
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

static	void	activateCB_pushButton37(
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

static	void	activateCB_pushButton39(
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

static	void	singleSelectionCB_scrolledList5(
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

static	void	defaultActionCB_scrolledList5(
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

static Widget	_Uxbuild_ScriptInstaller()
{
	Widget		_UxParent;


	/* Creation of ScriptInstaller */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	ScriptInstaller = XtVaCreatePopupShell( "ScriptInstaller",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 449,
			XmNheight, 280,
			XmNx, 429,
			XmNy, 406,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNiconName, "Ferret: Script Installer",
			XmNtitle, "Script Installer",
			NULL );


	/* Creation of scrolledWindowText7 */
	scrolledWindowText7 = XtVaCreateManagedWidget( "scrolledWindowText7",
			xmScrolledWindowWidgetClass,
			ScriptInstaller,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNx, 9,
			XmNy, 33,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNunitType, XmPIXELS,
			NULL );


	/* Creation of form31 */
	form31 = XtVaCreateManagedWidget( "form31",
			xmFormWidgetClass,
			scrolledWindowText7,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 0,
			XmNy, 0,
			XmNwidth, 418,
			XmNheight, 280,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of label34 */
	label34 = XtVaCreateManagedWidget( "label34",
			xmLabelWidgetClass,
			form31,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Scripts:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNx, 13,
			XmNy, 20,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of pushButton40 */
	pushButton40 = XtVaCreateManagedWidget( "pushButton40",
			xmPushButtonWidgetClass,
			form31,
			RES_CONVERT( XmNlabelString, "Done" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNx, 203,
			XmNy, 242,
			XmNleftPosition, 45,
			XmNleftOffset, 1,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNbottomOffset, 10,
			XmNbottomAttachment, XmATTACH_FORM,
			NULL );
	XtAddCallback( pushButton40, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton40,
		(XtPointer) NULL );



	/* Creation of scrolledWindowList2 */
	scrolledWindowList2 = XtVaCreateManagedWidget( "scrolledWindowList2",
			xmScrolledWindowWidgetClass,
			form31,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNshadowThickness, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNheight, 130,
			XmNx, 10,
			XmNy, 41,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 5,
			XmNtopWidget, label34,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNbottomOffset, 50,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNwidth, 139,
			NULL );


	/* Creation of scrolledList2 */
	scrolledList2 = XtVaCreateManagedWidget( "scrolledList2",
			xmListWidgetClass,
			scrolledWindowList2,
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNselectionPolicy, XmSINGLE_SELECT,
			XmNheight, 130,
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120*" ),
			XmNwidth, 133,
			NULL );
	XtAddCallback( scrolledList2, XmNsingleSelectionCallback,
		(XtCallbackProc) singleSelectionCB_scrolledList2,
		(XtPointer) NULL );
	XtAddCallback( scrolledList2, XmNdefaultActionCallback,
		(XtCallbackProc) defaultActionCB_scrolledList2,
		(XtPointer) NULL );



	/* Creation of rowColumn40 */
	rowColumn40 = XtVaCreateManagedWidget( "rowColumn40",
			xmRowColumnWidgetClass,
			form31,
			XmNx, 167,
			XmNy, 176,
			XmNwidth, 109,
			XmNheight, 85,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNleftOffset, 10,
			XmNleftWidget, scrolledWindowList2,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 50,
			XmNtopWidget, scrolledWindowList2,
			XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNbottomOffset, 50,
			XmNbottomWidget, scrolledWindowList2,
			XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET,
			NULL );


	/* Creation of pushButton37 */
	pushButton37 = XtVaCreateManagedWidget( "pushButton37",
			xmPushButtonWidgetClass,
			rowColumn40,
			XmNx, 13,
			XmNy, 6,
			XmNwidth, 90,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "Add >>" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( pushButton37, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton37,
		(XtPointer) NULL );



	/* Creation of pushButton39 */
	pushButton39 = XtVaCreateManagedWidget( "pushButton39",
			xmPushButtonWidgetClass,
			rowColumn40,
			XmNx, -2,
			XmNy, 3,
			XmNwidth, 90,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "Remove >>" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			NULL );
	XtAddCallback( pushButton39, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton39,
		(XtPointer) NULL );



	/* Creation of label52 */
	label52 = XtVaCreateManagedWidget( "label52",
			xmLabelWidgetClass,
			form31,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Items in Script Menu:" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNx, 250,
			XmNy, 17,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNwidth, 151,
			XmNleftOffset, 10,
			XmNleftWidget, rowColumn40,
			XmNleftAttachment, XmATTACH_WIDGET,
			NULL );


	/* Creation of scrolledWindowList5 */
	scrolledWindowList5 = XtVaCreateManagedWidget( "scrolledWindowList5",
			xmScrolledWindowWidgetClass,
			form31,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNshadowThickness, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNheight, 130,
			XmNx, 223,
			XmNy, 37,
			XmNtopOffset, 5,
			XmNtopWidget, label52,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftOffset, 10,
			XmNleftWidget, rowColumn40,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNbottomOffset, 50,
			XmNbottomAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of scrolledList5 */
	scrolledList5 = XtVaCreateManagedWidget( "scrolledList5",
			xmListWidgetClass,
			scrolledWindowList5,
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNselectionPolicy, XmSINGLE_SELECT,
			XmNheight, 130,
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120*" ),
			XmNwidth, 163,
			NULL );
	XtAddCallback( scrolledList5, XmNsingleSelectionCallback,
		(XtCallbackProc) singleSelectionCB_scrolledList5,
		(XtPointer) NULL );
	XtAddCallback( scrolledList5, XmNdefaultActionCallback,
		(XtCallbackProc) defaultActionCB_scrolledList5,
		(XtPointer) NULL );




	return ( ScriptInstaller );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_ScriptInstaller( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	rtrn = _Uxbuild_ScriptInstaller();

	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

