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
	Open_Save_jnl.c

       Associated Header file: Open_Save_jnl.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/FileSB.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/


/* globals */
swidget Save_jnl;
swidget gSavedSjnl = NULL;
extern swidget fileSelectionBox4;

extern void MMCancelSave(void);
extern void MMSaveAsOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Open_Save_jnl.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Save_jnl;
Widget	fileSelectionBox4;

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	cancelCB_fileSelectionBox4(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	MMCancelSave();
	}
}

static	void	okCallback_fileSelectionBox4(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	MMSaveAsOK(UxWidget, UxClientData, UxCallbackArg);
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Save_jnl()
{
	Widget		_UxParent;


	/* Creation of Save_jnl */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Save_jnl = XtVaCreatePopupShell( "Save_jnl",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 400,
			XmNheight, 298,
			XmNx, 493,
			XmNy, 282,
			XmNiconName, "Save Macro",
			RES_CONVERT( XmNbackground, "gray" ),
			XmNallowShellResize, TRUE,
			NULL );


	/* Creation of fileSelectionBox4 */
	fileSelectionBox4 = XtVaCreateManagedWidget( "fileSelectionBox4",
			xmFileSelectionBoxWidgetClass,
			Save_jnl,
			XmNwidth, 400,
			XmNheight, 298,
			XmNx, 32,
			XmNy, 28,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNforeground, "black" ),
			XmNlabelFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNbuttonFontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			XmNtextFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			NULL );
	XtAddCallback( fileSelectionBox4, XmNcancelCallback,
		(XtCallbackProc) cancelCB_fileSelectionBox4,
		(XtPointer) NULL );
	XtAddCallback( fileSelectionBox4, XmNokCallback,
		(XtCallbackProc) okCallback_fileSelectionBox4,
		(XtPointer) NULL );




	return ( Save_jnl );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Save_jnl( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedSjnl == NULL) {
		rtrn = _Uxbuild_Save_jnl();

		gSavedSjnl = rtrn;
		
			/* hide the help button */
		        XtUnmanageChild(XmFileSelectionBoxGetChild(UxGetWidget(fileSelectionBox4), XmDIALOG_HELP_BUTTON));
		}
		else
			rtrn = gSavedSjnl;
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

