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

