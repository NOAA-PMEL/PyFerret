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
	Open_Save_dset.c

       Associated Header file: Open_Save_dset.h
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

#include "ferret_structures.h"
#include "ferret.h"

#define charset XmSTRING_DEFAULT_CHARSET

/* globals */
swidget Open_Save_dset, FerretMainWd;
swidget gSavedOSdset = NULL;

/* prototypes */
extern void set_dset(Widget wid, XtPointer client_data, XtPointer call_data);
extern void cancel_dset(Widget wid, XtPointer client_data, XtPointer call_data);
extern void ferret_command(char *cmdText, int cmdMode);

static int x, y;


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Open_Save_dset.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Open_Save_dset;
Widget	fileSelectionBox1;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

void cancel_dset( Widget wid, XtPointer client_data, XtPointer call_data )
{
  	XtPopdown(UxGetWidget(Open_Save_dset));
}

void set_dset( Widget wid, XtPointer client_data, XtPointer call_data )
{
	char *pathname, dset[MAX_NAME_LENGTH]="", cmd[MAX_COMMAND_LENGTH]="";
  	char tempText[MAX_NAME_LENGTH]="";
	int i=0, pos=0, c=0;
	XmFileSelectionBoxCallbackStruct *cbs=(XmFileSelectionBoxCallbackStruct *) call_data;

  	if (!XmStringGetLtoR(cbs->value, charset, &pathname))
      		return; /* must have been an internal error */

  	if (!*pathname) { /* nothing typed? */
      		puts("No file selected.");
      		XtFree(pathname); 		/* even "" is an allocated byte */
      		XtPopdown(UxGetWidget(Open_Save_dset));
      		return;
    	}

	if (strlen(pathname) == 0) {
		/* nothing typed? */
      		puts("No file selected.");
      		XtFree(pathname); 		/* even "" is an allocated byte */
      		XtPopdown(UxGetWidget(Open_Save_dset));
      		return;
	}

	if (strstr(pathname, "/")) {
		/* isolate the data set name from path name */
		for (i=strlen(pathname); i>=0; i--) {
			if (pathname[i-1] == '/') {
				pos = i;
			break;
			}
		}

		c = 0;
		for (i=pos; i<strlen(pathname); i++)
			dset[c++] = pathname[i];
		dset[c] = '\0';
	}
	else
		/* no path name, just a dataset name */
		strcpy(dset, pathname);

/*
 * - Convert dset name to upper case.
 * - If dset ends in ".cdf", remove it.
 * - Issue the "SET DATA" command to Ferret.
 * - Dismiss the file selection box.
 */
	strcpy(tempText, dset);
	for (i=0; i<strlen(tempText); i++)
		tempText[i] = toupper(tempText[i]);
	if (strstr(tempText, ".CDF"))
		dset[strlen(dset)-4] = '\0';
	sprintf(cmd, "SET DATA \"%s\"", pathname);
	ferret_command(cmd, IGNORE_COMMAND_WIDGET);
 	XtPopdown(UxGetWidget(Open_Save_dset));

}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	destroyCB_Open_Save_dset(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedOSdset = NULL;
}

static	void	cancelCB_fileSelectionBox1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	cancel_dset(UxWidget, UxClientData, UxCallbackArg);
	}
}

static	void	okCallback_fileSelectionBox1(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	set_dset(UxWidget, UxClientData, UxCallbackArg);
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Open_Save_dset()
{
	Widget		_UxParent;


	/* Creation of Open_Save_dset */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Open_Save_dset = XtVaCreatePopupShell( "Open_Save_dset",
			topLevelShellWidgetClass,
			_UxParent,
			RES_CONVERT( XmNiconName, "Ferret: Open Dataset" ),
			RES_CONVERT( XmNtitle, "Open Dataset" ),
			NULL );
	XtAddCallback( Open_Save_dset, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_Open_Save_dset,
		(XtPointer) NULL );



	/* Creation of fileSelectionBox1 */
	fileSelectionBox1 = XtVaCreateManagedWidget( "fileSelectionBox1",
			xmFileSelectionBoxWidgetClass,
			Open_Save_dset,
			RES_CONVERT( XmNdialogTitle, "Open Ferret Data Set" ),
			NULL );
	XtAddCallback( fileSelectionBox1, XmNcancelCallback,
		(XtCallbackProc) cancelCB_fileSelectionBox1,
		(XtPointer) NULL );
	XtAddCallback( fileSelectionBox1, XmNokCallback,
		(XtCallbackProc) okCallback_fileSelectionBox1,
		(XtPointer) NULL );




	return ( Open_Save_dset );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Open_Save_dset( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedOSdset == NULL) {
		rtrn = _Uxbuild_Open_Save_dset();

		gSavedOSdset = rtrn;
		
			/* hide the help button */
		        XtUnmanageChild(XmFileSelectionBoxGetChild(UxGetWidget(fileSelectionBox1), XmDIALOG_HELP_BUTTON));
		}
		else
			rtrn = gSavedOSdset;
		
		XtVaGetValues(UxGetWidget(FerretMainWd),
			XmNx, &x,
			XmNy, &y,
			NULL);
		
		XtVaSetValues(UxGetWidget(rtrn),
			XmNx, x,
			XmNy, y,
			NULL);
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

