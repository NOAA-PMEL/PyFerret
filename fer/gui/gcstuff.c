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
