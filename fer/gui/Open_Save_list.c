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
	Open_Save_list.c

       Associated Header file: Open_Save_list.h
*******************************************************************************/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
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

#include <sys/types.h>
#include <sys/stat.h>

swidget create_Open_Save_list(swidget UxParent);

extern void ListSaveAsOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void ListCancel(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void SaveAsListFile(void);


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Open_Save_list.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Open_Save_list;
Widget	fileSelectionBox6;

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	applyCB_fileSelectionBox6(
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

static	void	cancelCB_fileSelectionBox6(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ListCancel(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	okCallback_fileSelectionBox6(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	ListSaveAsOK(UxWidget, UxClientData, UxCallbackArg);
	
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Open_Save_list()
{
	Widget		_UxParent;


	/* Creation of Open_Save_list */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Open_Save_list = XtVaCreatePopupShell( "Open_Save_list",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 382,
			XmNheight, 358,
			XmNx, 400,
			XmNy, 272,
			XmNallowShellResize, TRUE,
			NULL );


	/* Creation of fileSelectionBox6 */
	fileSelectionBox6 = XtVaCreateManagedWidget( "fileSelectionBox6",
			xmFileSelectionBoxWidgetClass,
			Open_Save_list,
			XmNwidth, 382,
			XmNheight, 358,
			XmNx, 19,
			XmNy, 7,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNforeground, "black" ),
			XmNlabelFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNbuttonFontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			XmNtextFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			NULL );
	XtAddCallback( fileSelectionBox6, XmNapplyCallback,
		(XtCallbackProc) applyCB_fileSelectionBox6,
		(XtPointer) NULL );
	XtAddCallback( fileSelectionBox6, XmNcancelCallback,
		(XtCallbackProc) cancelCB_fileSelectionBox6,
		(XtPointer) NULL );
	XtAddCallback( fileSelectionBox6, XmNokCallback,
		(XtCallbackProc) okCallback_fileSelectionBox6,
		(XtPointer) NULL );




	return ( Open_Save_list );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Open_Save_list( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	rtrn = _Uxbuild_Open_Save_list();

	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

