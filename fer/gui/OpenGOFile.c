
/*******************************************************************************
	OpenGOFile.c

       Associated Header file: OpenGOFile.h
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
#define charset XmSTRING_DEFAULT_CHARSET

/* variables */
swidget OpenGOFile;

static GOCancelOpen(void);
static GOOpenOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void ferret_command(char *cmdText, int cmdMode);
swidget create_OpenGOFile(swidget UxParent);


static	Widget	fileSelectionBox9;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "OpenGOFile.h"
#undef CONTEXT_MACRO_ACCESS

Widget	OpenGOFile;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static GOCancelOpen()
{
	/* dismiss the file selection box */
  	XtDestroyWidget(UxGetWidget(OpenGOFile));
}

static GOOpenOK(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	/* callback from data set selector widget */
	char *pathname, go[32], cmd[48], quote='"';
	int i, pos, c;
	XmFileSelectionBoxCallbackStruct *cbs;

	cbs = (XmFileSelectionBoxCallbackStruct *)UxCallbackArg;

  	if (!XmStringGetLtoR(cbs->value, charset, &pathname))
      		return; /* must have been an internal error */

  	if (!*pathname) { /* nothing typed? */
      		puts("No file selected.");
      		XtFree(pathname); 		/* even "" is an allocated byte */
      		XtDestroyWidget(UxGetWidget(OpenGOFile));
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
			go[c++] = pathname[i];
		go[c] = '\0';

		if (strlen(go) > 0)
			strcpy(go, pathname);
	}
	else
		/* no path name, just a go name--must be in the ferret paths */
		strcpy(go, pathname);

	if (strlen(go) == 0) {
		/* nothing typed? */
      		puts("No file selected.");
      		XtFree(pathname);
      		XtDestroyWidget(UxGetWidget(OpenGOFile));
      		return;
	}
	
	/* send go cmd to ferret */
	sprintf(cmd, "GO %c%s%c", quote, go, quote);
	ferret_command(cmd, IGNORE_COMMAND_WIDGET);
  	XtDestroyWidget(UxGetWidget(OpenGOFile));
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	cancelCB_fileSelectionBox9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	GOCancelOpen();
}

static	void	okCallback_fileSelectionBox9(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	GOOpenOK(UxWidget, UxClientData, UxCallbackArg);
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_OpenGOFile()
{
	Widget		_UxParent;


	/* Creation of OpenGOFile */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	OpenGOFile = XtVaCreatePopupShell( "OpenGOFile",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 317,
			XmNheight, 366,
			XmNx, 374,
			XmNy, 311,
			XmNiconName, "Ferret: Open GO File",
			XmNtitle, "Ferret Open GO File",
			XmNallowShellResize, TRUE,
			NULL );


	/* Creation of fileSelectionBox9 */
	fileSelectionBox9 = XtVaCreateManagedWidget( "fileSelectionBox9",
			xmFileSelectionBoxWidgetClass,
			OpenGOFile,
			XmNwidth, 39,
			XmNheight, 84,
			XmNx, 260,
			XmNy, 27,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNbuttonFontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNlabelFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			XmNtextFontList, UxConvertFontList( "*courier-medium-r-*-120-*" ),
			NULL );
	XtAddCallback( fileSelectionBox9, XmNcancelCallback,
		(XtCallbackProc) cancelCB_fileSelectionBox9,
		(XtPointer) NULL );
	XtAddCallback( fileSelectionBox9, XmNokCallback,
		(XtCallbackProc) okCallback_fileSelectionBox9,
		(XtPointer) NULL );




	return ( OpenGOFile );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_OpenGOFile( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	rtrn = _Uxbuild_OpenGOFile();

	/* hide the help button */
	XtUnmanageChild(XmFileSelectionBoxGetChild(UxGetWidget(fileSelectionBox9), XmDIALOG_HELP_BUTTON));
	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

