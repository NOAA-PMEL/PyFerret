
/*******************************************************************************
	PixmapStuff.c

       Associated Header file: PixmapStuff.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "xpm.h"

/* prototypes */
Pixmap GetPixmapFromData(char **inData);
Pixmap GetPixmapFromFile(char *inFile);
Pixmap GetBitmapFromData(char *inData, unsigned int width, unsigned int height);

/* globals */
extern swidget FerretMainWd, Splash;
extern Boolean gFerretIsStarting;


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "PixmapStuff.h"
#undef CONTEXT_MACRO_ACCESS

Widget	PixmapStuff;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

Pixmap GetPixmapFromData(inData)
char **inData;
{
	Display *display;
	Screen *screen;
	Pixmap rtnPix, rtnMask;
	Window wd;
	int err;
 
	display = XtDisplay((UxGetWidget(FerretMainWd)));
	screen = DefaultScreenOfDisplay(display);
	wd = RootWindowOfScreen(screen);
	err = XpmCreatePixmapFromData(display, wd, inData, &rtnPix, &rtnMask, NULL);
	return rtnPix;
}

Pixmap GetPixmapFromFile(inFile)
char *inFile;
{
	Display *display;
	Screen *screen;
	Pixmap rtnPix, rtnMask;
	int err;
	Window wd;
 
	display = XtDisplay((UxGetWidget(FerretMainWd)));
	screen = DefaultScreenOfDisplay(display);
	wd = RootWindowOfScreen(screen);
	err = XpmReadFileToPixmap(display, wd, inFile, &rtnPix, &rtnMask, NULL);
	return rtnPix;
}

Pixmap GetBitmapFromData(inData, width, height)
char *inData;
unsigned int width, height;
{
	Display *display;
	Screen *screen;
	Window wd;
	int err;
	Pixmap rtnPix;
 
	display = XtDisplay((UxGetWidget(FerretMainWd)));
	screen = DefaultScreenOfDisplay(display);
	wd = RootWindowOfScreen(screen);
	rtnPix = XCreateBitmapFromData(display, wd, inData, width, height);
	return rtnPix;
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_PixmapStuff()
{
	Widget		_UxParent;


	/* Creation of PixmapStuff */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	PixmapStuff = XtVaCreatePopupShell( "PixmapStuff",
			topLevelShellWidgetClass,
			_UxParent,
			XmNx, 660,
			XmNy, 123,
			XmNwidth, 333,
			XmNheight, 396,
			NULL );



	return ( PixmapStuff );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_PixmapStuff( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	rtrn = _Uxbuild_PixmapStuff();

	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

