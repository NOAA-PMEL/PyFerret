
/*******************************************************************************
	Open_Save_list.c

       Associated Header file: Open_Save_list.h
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

