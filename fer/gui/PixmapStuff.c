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

