
/*******************************************************************************
	Open_Save_grd.c

       Associated Header file: Open_Save_grd.h
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
swidget Open_Save_grd;
swidget gSavedOSgrd = NULL;
extern swidget fileSelectionBox3;


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Open_Save_grd.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Open_Save_grd;
Widget	fileSelectionBox3;

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Open_Save_grd()
{
	Widget		_UxParent;


	/* Creation of Open_Save_grd */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Open_Save_grd = XtVaCreatePopupShell( "Open_Save_grd",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 400,
			XmNheight, 355,
			XmNx, 455,
			XmNy, 203,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNallowShellResize, TRUE,
			NULL );


	/* Creation of fileSelectionBox3 */
	fileSelectionBox3 = XtVaCreateManagedWidget( "fileSelectionBox3",
			xmFileSelectionBoxWidgetClass,
			Open_Save_grd,
			XmNwidth, 164,
			XmNheight, 213,
			XmNx, 14,
			XmNy, 38,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNforeground, "black" ),
			XmNlabelFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNbuttonFontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			XmNtextFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			NULL );



	return ( Open_Save_grd );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Open_Save_grd( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedOSgrd == NULL) {
		rtrn = _Uxbuild_Open_Save_grd();

		gSavedOSgrd = rtrn;
		
			/* hide the help button */
		        XtUnmanageChild(XmFileSelectionBoxGetChild(UxGetWidget(fileSelectionBox3), XmDIALOG_HELP_BUTTON));
		}
		else
			rtrn = gSavedOSgrd;
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

