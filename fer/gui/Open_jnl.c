
/*******************************************************************************
	Open_jnl.c

       Associated Header file: Open_jnl.h
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
swidget Open_jnl;
swidget gSavedOjnl = NULL;
extern swidget fileSelectionBox5;

extern void MMCancelOpen(void);
extern void MMOpenOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Open_jnl.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Open_jnl;
Widget	fileSelectionBox5;

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	applyCB_fileSelectionBox5(
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

static	void	cancelCB_fileSelectionBox5(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	MMCancelOpen();
	}
}

static	void	okCallback_fileSelectionBox5(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	MMOpenOK(UxWidget, UxClientData, UxCallbackArg);
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Open_jnl()
{
	Widget		_UxParent;


	/* Creation of Open_jnl */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Open_jnl = XtVaCreatePopupShell( "Open_jnl",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 400,
			XmNheight, 321,
			XmNx, 503,
			XmNy, 269,
			XmNiconName, "Open Macro",
			RES_CONVERT( XmNbackground, "gray" ),
			XmNallowShellResize, TRUE,
			NULL );


	/* Creation of fileSelectionBox5 */
	fileSelectionBox5 = XtVaCreateManagedWidget( "fileSelectionBox5",
			xmFileSelectionBoxWidgetClass,
			Open_jnl,
			XmNwidth, 131,
			XmNheight, 234,
			XmNx, 32,
			XmNy, 28,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNforeground, "black" ),
			XmNlabelFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNbuttonFontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			XmNtextFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			NULL );
	XtAddCallback( fileSelectionBox5, XmNapplyCallback,
		(XtCallbackProc) applyCB_fileSelectionBox5,
		(XtPointer) NULL );
	XtAddCallback( fileSelectionBox5, XmNcancelCallback,
		(XtCallbackProc) cancelCB_fileSelectionBox5,
		(XtPointer) NULL );
	XtAddCallback( fileSelectionBox5, XmNokCallback,
		(XtCallbackProc) okCallback_fileSelectionBox5,
		(XtPointer) NULL );




	return ( Open_jnl );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Open_jnl( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedOjnl == NULL) {
		rtrn = _Uxbuild_Open_jnl();

		gSavedOjnl = rtrn;
		
			/* hide the help button */
		        XtUnmanageChild(XmFileSelectionBoxGetChild(UxGetWidget(fileSelectionBox5), XmDIALOG_HELP_BUTTON));
		}
		else
			rtrn = gSavedOjnl;
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

