/* grab_image_xwd.c - containing 

         void Window_Dump(window, display, outfilename, file_type);

 NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

 Nov. '94 - Kevin O'Brien based on xwd code from MIT
 Dec. '94 - *kob* update to allow writing out of HDF 8 bit raster files
 May. '95 - *kob* added ifdef's for hp port
 Sep. '95 - *kob* added ifdef for sgi port...had to change the signal 
                  handling because of an error that occurred on sgi when
		  trying to write out a gif file
 Oct. '95 - *sh*  error messages were being sent to stdout instead of stderr
                  In Web server this caused them to be overlooked
   compile with
        cc -g -c -I/home/rogue/hankin/fer/src/common grab_image_xwd.c 
   on sgi with
        cc -DSGI_SIGNALS -g -c -I/home/rogue/hankin/fer/src/common grab_image_xwd.c 

 May. '96 - *kob* added free commands for arrays r,g,b

*/ 
   
/* $XConsortium: xwd.c,v 1.56 91/07/25 18:00:15 rws Exp $ */

/* Copyright 1987 Massachusetts Institute of Technology */

/*
 * xwd.c MIT Project Athena, X Window system window raster image dumper.
 *
 * This program will dump a raster image of the contents of a window into a 
 * file for output on graphics printers or for other uses.
 *
 *  Author:	Tony Della Fera, DEC
 *		17-Jun-85
 * 
 *  Modification history:
 *
 *  11/14/86 Bill Wyatt, Smithsonian Astrophysical Observatory
 *    - Removed Z format option, changing it to an XY option. Monochrome 
 *      windows will always dump in XY format. Color windows will dump
 *      in Z format by default, but can be dumped in XY format with the
 *      -xy option.
 *
 *  11/18/86 Bill Wyatt
 *    - VERSION 6 is same as version 5 for monchrome. For colors, the 
 *      appropriate number of Color structs are dumped after the header,
 *      which has the number of colors (=0 for monochrome) in place of the
 *      V5 padding at the end. Up to 16-bit displays are supported. I
 *      don't yet know how 24- to 32-bit displays will be handled under
 *      the Version 11 protocol.
 *
 *  6/15/87 David Krikorian, MIT Project Athena
 *    - VERSION 7 runs under the X Version 11 servers, while the previous
 *      versions of xwd were are for X Version 10.  This version is based
 *      on xwd version 6, and should eventually have the same color
 *      abilities. (Xwd V7 has yet to be tested on a color machine, so
 *      all color-related code is commented out until color support
 *      becomes practical.)
 *
 *   4/5/94 Achille Hui,
 *    - patches for alpha to write out the XColor structures in a more
 *	portable manner. Notice the XWDColor structure is defined only
 *	in /usr/include/X11/XWDFile.h on an alpha running OSF/1.
 */

/*%
 *%    This is the format for commenting out color-related code until
 *%  color can be supported.
%*/

#include <stdio.h>
#include <errno.h>

#include <X11/Xos.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

/*add include for signal for sunOS benefit *kob*/
#include <signal.h>

#ifdef NO_WIN_UTIL_H
#else
#include <X11/Xmu/WinUtil.h>
#endif

typedef unsigned long Pixel;

#define FEEP_VOLUME 0

/* Include routines to do parsing */
#include "dsimple.h"

/* Setable Options */

int format = ZPixmap;
Bool nobdrs = False;
Bool on_root = False;
Bool standard_out = True;
Bool debug = False;
Bool use_installed = False;
long add_pixel_value = 0;

extern int (*_XErrorFunction)();
extern int _XDefaultError();


static long parse_long (s)
    char *s;
{
    char *fmt = "%lu";
    long retval = 0L;
    int thesign = 1;

    if (s && s[0]) {
	if (s[0] == '-') s++, thesign = -1;
	if (s[0] == '0') s++, fmt = "%lo";
	if (s[0] == 'x' || s[0] == 'X') s++, fmt = "%lx";
	(void) sscanf (s, fmt, &retval);
    }
    return (thesign * retval);
}

/*
 * Window_Dump: dump a window to a file which must already be open for
 *              writting.
 */

char *calloc();

/* #include "X11/XWDFile.h" */

Window_Dump(window, dpy,outfile, type)
     Window window;
     Display *dpy;
     char *outfile;
     char *type;
{
    unsigned long swaptest = 1;
    XColor *colors;
    unsigned buffer_size;
    int win_name_size;
    int header_size;
    int ncolors, i;
    char *win_name;
    Bool got_win_name;
    XWindowAttributes win_info;
    XImage *image;
    int absx, absy, x, y;
    unsigned width, height;
    int dwidth, dheight;
    int bw;
    Window dummywin;
    int *r, *g, *b;
    void (*func)();
    Pixmap pixmap;
    FILE *fp;

#ifdef HP_SIGNALS
    func = signal(_SIGIO, SIG_DFL);
#elif SGI_SIGNALS
    func = signal(SIGIO, SIG_IGN);
#else
    func = signal(SIGIO, SIG_DFL);
#endif



    /*
     * Inform the user not to alter the screen.
     */
/*    Beep(); */

    /*
     * Get the parameters of the window being dumped.
     */
    if(!XGetWindowAttributes(dpy, window, &win_info)) 
      {
	fprintf (stderr, "Can't get target window attributes.");
#ifdef HP_SIGNALS
	signal(_SIGIO, func);
#else
	signal(SIGIO, func);
#endif
	exit(1);
      }

    /* handle any frame window */
    if (!XTranslateCoordinates (dpy, window, RootWindow (dpy, screen), 0, 0,
				&absx, &absy, &dummywin)) {
	fprintf (stderr, 
		 "%s:  unable to translate window coordinates (%d,%d)\n",
		 program_name, absx, absy);
#ifdef HP_SIGNALS
	signal(_SIGIO, func);
#else
	signal(SIGIO, func);
#endif	
	exit (1);
    }
    win_info.x = absx;
    win_info.y = absy;
    width = win_info.width;
    height = win_info.height;
    bw = 0;

/* If borders are taken into account with unmapped windows, things crash. 
   this check on borders in xwd code not needed here.  
   if (!nobdrs) {
	absx -= win_info.border_width;
	absy -= win_info.border_width;
	bw = win_info.border_width;
	width += (2 * bw);
	height += (2 * bw);
	}

*/
    dwidth = DisplayWidth (dpy, screen);
    dheight = DisplayHeight (dpy, screen);


    /* clip to window */
    if (absx < 0) width += absx, absx = 0;
    if (absy < 0) height += absy, absy = 0;
    if (absx + width > dwidth) width = dwidth - absx;
    if (absy + height > dheight) height = dheight - absy;

    XFetchName(dpy, window, &win_name);
    if (!win_name || !win_name[0]) {
	win_name = "xwdump";
	got_win_name = False;
    } else {
	got_win_name = True;
    }

    /* sizeof(char) is included for the null string terminator. */
    win_name_size = strlen(win_name) + sizeof(char);

    /*
     * Snarf the pixmap with XGetImage.
     */

    x = absx - win_info.x;
    y = absy - win_info.y;
    if (on_root)
	image = XGetImage (dpy, RootWindow(dpy, screen), absx, absy, width, height, AllPlanes, format);
    else if (win_info.map_state == IsUnmapped) {
/* create and use pixmap to generate image if window is unmapped  *kob*/
      pixmap = XCreatePixmap(dpy, window, width, height, win_info.depth);
      XCopyArea(dpy, window, pixmap, DefaultGC(dpy, DefaultScreen(dpy)),
		x, y, width, height, 0, 0);
      image = XGetImage(dpy, pixmap, 0, 0, width, height,
			AllPlanes,ZPixmap);
      XFreePixmap(dpy, pixmap);
    }
    else
      image = XGetImage (dpy, window, x, y, width, height, AllPlanes, format);

    if (!image) {
	fprintf (stderr, "%s:  unable to get image at %dx%d+%d+%d\n",
		 program_name, width, height, x, y);
#ifdef HP_SIGNALS
	signal(_SIGIO, func);
#else
	signal(SIGIO, func);
#endif
	return;
    }

    if (add_pixel_value != 0) XAddPixel (image, add_pixel_value);

    /*
     * Determine the pixmap size.
     */
    buffer_size = Image_Size(image);

/*     if (debug) outl("xwd: Getting Colors.\n");*/

    ncolors = Get_XColors(&win_info, &colors,dpy);

#ifdef HP_SIGNALS
    signal(_SIGIO, func);
#else
    signal(SIGIO, func);
#endif
    /*
     * Inform the user that the image has been retrieved.
     */
#ifdef BELL
    XBell(dpy, FEEP_VOLUME);
    XBell(dpy, FEEP_VOLUME);
    XFlush(dpy);
#endif

    r = (int *)malloc(sizeof(int) * ncolors);
    g = (int *)malloc(sizeof(int) * ncolors);
    b = (int *)malloc(sizeof(int) * ncolors); 
    for (i=0; i < ncolors; i++) {
      r[i] = colors[i].red;
      g[i] = colors[i].green;
      b[i] = colors[i].blue;
    }



    if (strcmp(type, "GIF") == 0)
      if ((fp = fopen(outfile, "w")) == NULL) 
	fprintf (stderr,
		 "\nwrite_gif: can't open output file %s\n", outfile);
      else 
	wGIF(fp,image,r,g,b);
    else 
      wHDF(outfile,image,r,g,b);
    

/*    if(debug && ncolors > 0) outl("xwd: Freeing colors.\n"); */
/* *kob* 5/96 - also free the arrays r,g,b */
    if(ncolors > 0) free(colors);free(r); free(g); free(b);

    /*
     * Free window name string.
     */
/*    if (debug) outl("xwd: Freeing window name string.\n"); */
    if (got_win_name) XFree(win_name);

    /*
     * Free image
     */
    XDestroyImage(image);
}

/*
 * Report the syntax for calling xwd.
 */
/*usage()
{
    fprintf (stderr,
"usage: %s [-display host:dpy] [-debug] [-help] %s [-nobdrs] [-out <file>]",
	   program_name, SELECT_USAGE);
    fprintf (stderr, " [-kludge] [-xy] [-add value] [-frame]\n");
    exit(1);
}
*/

/*
 * Error - Fatal xwd error.
 */
extern int errno;

Error(string)
	char *string;	/* Error description string. */
{
/*	outl("\nxwd: Error => %s\n", string); */
	if (errno != 0) {
		perror("xwd");
	/*outl("\n"); */
	} exit(1);
}


/*
 * Determine the pixmap size.
 */

int Image_Size(image)
     XImage *image;
{
    if (image->format != ZPixmap)
      return(image->bytes_per_line * image->height * image->depth);

    return(image->bytes_per_line * image->height);
}

#define lowbit(x) ((x) & (~(x) + 1))

/*
 * Get the XColors of all pixels in image - returns # of colors
 */
int Get_XColors(win_info, colors,dpy)
     XWindowAttributes *win_info;
     XColor **colors;
     Display *dpy;
{
    int i, ncolors;
    Colormap cmap = win_info->colormap;

    if (use_installed)
	/* assume the visual will be OK ... */
	cmap = XListInstalledColormaps(dpy, win_info->root, &i)[0];
    if (!cmap)
	return(0);

    ncolors = win_info->visual->map_entries;
    if (!(*colors = (XColor *) malloc (sizeof(XColor) * ncolors)))
      {
	fprintf (stderr, "Fatal Error - Out of memory!");
	exit (1);
      }

    if (win_info->visual->class == DirectColor ||
	win_info->visual->class == TrueColor) {
	Pixel red, green, blue, red1, green1, blue1;

	red = green = blue = 0;
	red1 = lowbit(win_info->visual->red_mask);
	green1 = lowbit(win_info->visual->green_mask);
	blue1 = lowbit(win_info->visual->blue_mask);
	for (i=0; i<ncolors; i++) {
	  (*colors)[i].pixel = red|green|blue;
	  (*colors)[i].pad = 0;
	  red += red1;
	  if (red > win_info->visual->red_mask)
	    red = 0;
	  green += green1;
	  if (green > win_info->visual->green_mask)
	    green = 0;
	  blue += blue1;
	  if (blue > win_info->visual->blue_mask)
	    blue = 0;
	}
    } else {
	for (i=0; i<ncolors; i++) {
	  (*colors)[i].pixel = i;
	  (*colors)[i].pad = 0;
	}
    }

    XQueryColors(dpy, cmap, *colors, ncolors);
    
    return(ncolors);
}

_swapshort (bp, n)
    register char *bp;
    register unsigned n;
{
    register char c;
    register char *ep = bp + n;

    while (bp < ep) {
	c = *bp;
	*bp = *(bp + 1);
	bp++;
	*bp++ = c;
    }
}

_swaplong (bp, n)
    register char *bp;
    register unsigned n;
{
    register char c;
    register char *ep = bp + n;
    register char *sp;

    while (bp < ep) {
	sp = bp + 3;
	c = *sp;
	*sp = *bp;
	*bp++ = c;
	sp = bp + 1;
	c = *sp;
	*sp = *bp;
	*bp++ = c;
	bp += 2;
    }
}
