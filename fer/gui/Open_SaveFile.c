
/*******************************************************************************
	Open_SaveFile.c

       Associated Header file: Open_SaveFile.h
*******************************************************************************/

#include <stdio.h>
#include "UxLib.h"
#include "UxFsBox.h"
#include "UxTopSh.h"

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/


/* globals */
swidget Open_Save_dset;
swidget gSavedOSdset = NULL;


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Open_SaveFile.h"
#undef CONTEXT_MACRO_ACCESS

swidget	Open_Save_dset;
swidget	fileSelectionBox1;

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the swidgets and X widgets,
       and sets their properties to the values specified in the
       Property Editor.
*******************************************************************************/

static swidget	_Uxbuild_Open_Save_dset()
{
	/* Create the swidgets */


	/* Creation of Open_Save_dset */
	Open_Save_dset = UxCreateTopLevelShell( "Open_Save_dset", UxParent );

	UxPutWidth( Open_Save_dset, 273 );
	UxPutHeight( Open_Save_dset, 385 );
	UxPutX( Open_Save_dset, 716 );
	UxPutY( Open_Save_dset, 282 );
	UxCreateWidget( Open_Save_dset );


	/* Creation of fileSelectionBox1 */
	fileSelectionBox1 = UxCreateFileSelectionBox( "fileSelectionBox1", Open_Save_dset );
	UxPutWidth( fileSelectionBox1, 184 );
	UxPutHeight( fileSelectionBox1, 274 );
	UxPutX( fileSelectionBox1, 46 );
	UxPutY( fileSelectionBox1, 18 );
	UxPutUnitType( fileSelectionBox1, "pixels" );
	UxPutHelpLabelString( fileSelectionBox1, "Help" );
	UxCreateWidget( fileSelectionBox1 );



	/* UxRealizeInterface creates the X windows for the widgets above. */

	UxRealizeInterface( Open_Save_dset );

	return ( Open_Save_dset );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

swidget	create_Open_Save_dset( swidget _UxUxParent )
{
	swidget                 rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedOSdset == NULL) {
		rtrn = _Uxbuild_Open_Save_dset();

		gSavedOSdset = rtrn;
		}
		else
			rtrn = gSavedOSdset;
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

