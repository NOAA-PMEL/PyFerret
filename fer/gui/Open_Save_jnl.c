
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

