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



InitFerretStructs();
	
	display = XtDisplay(UxGetWidget(drawingArea1));
	screen = DefaultScreenOfDisplay(display);
	window = RootWindowOfScreen(screen);

	/* get the resolution of the device */
	widthMM = DisplayWidthMM(display, DefaultScreen(display));
	widthPix = DisplayWidth(display, DefaultScreen(display));
	widthIN = (float)widthMM/25.4;
	screenRez = (float)widthPix/widthIN;

	if (screenRez >= 91)
		gHiRez = True;
	else
		gHiRez = False;

	/* install a graphics context into user data of widget */ 
	gcv.foreground = WhitePixelOfScreen(screen);
	gcv.line_width = 1;
	gcv.background =;
	gcv.fill_style = FillOpaqueStippled;
	gcv.fill_rule = WindingRule;
	gcv.stipple = XCreatePixmapFromBitmapData(display, drawable,
		stippleData, 3, 3, fg, bg, 1);
				
	gc = XCreateGC(display, window, GCForeground+GCLineWidth, &gcv);
	XtVaSetValues(UxGetWidget(drawingArea1),
		XmNuserData, gc, 
		NULL);

	InitGlobalWidgets();
	InitPixmaps();

XtPopup(UxGetWidget(rtrn), no_grab);

/* set the hi rez size */
if (gHiRez) {
	Dimension width, height;
		
	XtVaGetValues(UxGetWidget(form1),
		XmNwidth, &width,
		XmNheight, &height,
		NULL);
	width = 1.15 * width;
	height = 1.2 * height;
		
	XtVaSetValues(UxGetWidget(form1),
		XmNwidth, width,
		XmNheight, height,
		NULL);
}
return(rtrn);
