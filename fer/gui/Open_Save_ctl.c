
/*******************************************************************************
	Open_Save_ctl.c

       Associated Header file: Open_Save_ctl.h
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

#include <Xm/FileSB.h>

/* globals */
swidget Open_Save_ctl;
swidget gSavedOSctl = NULL;


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Open_Save_ctl.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Open_Save_ctl;
Widget	fileSelectionBox2;

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Open_Save_ctl()
{
	Widget		_UxParent;


	/* Creation of Open_Save_ctl */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Open_Save_ctl = XtVaCreatePopupShell( "Open_Save_ctl",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 400,
			XmNheight, 359,
			XmNx, 442,
			XmNy, 214,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNallowShellResize, TRUE,
			NULL );


	/* Creation of fileSelectionBox2 */
	fileSelectionBox2 = XtVaCreateManagedWidget( "fileSelectionBox2",
			xmFileSelectionBoxWidgetClass,
			Open_Save_ctl,
			XmNwidth, 172,
			XmNheight, 309,
			XmNx, 27,
			XmNy, 29,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNforeground, "black" ),
			XmNlabelFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNbuttonFontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			XmNtextFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			NULL );



	return ( Open_Save_ctl );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Open_Save_ctl( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		Widget helpWID = NULL;
		
		if (gSavedOSctl == NULL) {
		rtrn = _Uxbuild_Open_Save_ctl();

		gSavedOSctl = rtrn;
		
			/* remove the help button */
			helpWID = XmFileSelectionBoxGetChild(fileSelectionBox2, XmDIALOG_HELP_BUTTON);
			if (helpWID)
				XtUnmanageChild(helpWID);
		}
		else
			rtrn = gSavedOSctl;
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

