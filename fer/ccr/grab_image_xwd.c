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

#include <unistd.h>
#include <stdio.h>
#include <errno.h>
/* #include <endian.h> */

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

    ncolors = Get_XColors(&win_info, image, &colors,dpy); 

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

    /*     *kob* 6/02 - if we are on a TrueColor or DirectColor display
	                we need to reset the image information to 
			what it was before we snapped the gif so that
			all of the memory is freed properly
     */
    if (win_info.visual->class == DirectColor ||
	win_info.visual->class == TrueColor) {
      image->bytes_per_line = image->bytes_per_line * 4; 
      image->bits_per_pixel = 32;
      image->depth          = 24;
    }
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
#define lowbyte(x) ((x) & (~(x) + 8))

/*
 * Get the XColors of all pixels in image - returns # of colors
 */
int Get_XColors(win_info, image, colors,dpy) 
     XImage *image;  
     XWindowAttributes *win_info;
     XColor **colors;
     Display *dpy;
{
    int i, ncolors;
    unsigned long pixel;
    unsigned char *cptr,tmp_cptr;

    Bool reverse_bytes;

    Colormap cmap = win_info->colormap;

    if (use_installed)
	/* assume the visual will be OK ... */
	cmap = XListInstalledColormaps(dpy, win_info->root, &i)[0];
    if (!cmap)
	return(0);

   /* in Ferret, which uses GKS, the color model is always indexed 8 bit */
    /* ncolors = win_info->visual->map_entries;*/
    ncolors = 256;

    if (!(*colors = (XColor *) malloc (sizeof(XColor) * ncolors)))
      {
	fprintf (stderr, "Fatal Error - Out of memory!");
	exit (1);
      }

    if (win_info->visual->class == DirectColor ||
	win_info->visual->class == TrueColor) {

	int nunique_colors;
	char ind; 
	char *color_indices =  (char *) image->data;
	unsigned int *color_values = (unsigned int *) image->data;
	/*	int npixels = image->bytes_per_line * image->height; */
	int npixels = image->width * image->height;

	/*
	  For TrueColor or DirectColor the colors are stored in
	  the image structure, rather than in a separate map. Here we convert the
	  TrueColor representation to an indexed color representation by pulling
	  the colors from the image and storing them in a color map, replacing
	  the colors with indices pointing to the color map.
	  
	  Since the indices are 1 byte, whereas the color pixels are 4, only 1/4
	  of the inage data is actually over-written.  At the end we need to modify the
	  image structure to reflect its new contents.
	  
	  Note: RISK OF MEMORY LEAK, IF X USES THESE VALS FREEING MEMORY
	  
	*/

        /* initialize unique color list to first color in image*/
	for (i=1; i<ncolors; i++) {
	  (*colors)[i].pixel = 0;
	  (*colors)[i].pad = 0;
	}
	
	(*colors)[0].pixel = color_values[0];
	nunique_colors = 1;

        /* convert direct color representation to indexed color */ 
	for (i=0; i<npixels; i++) {

	  /* see if this pixel matches a known color index */
	  ind = 0;

	  while ( (ind < (int) nunique_colors) 
               && (color_values[i] != (*colors)[ind].pixel) ) ind++;

	  /* store unique color just found */
	  if (ind == (int)nunique_colors ) {
	    (*colors)[ind].pixel = color_values[i]; 
	    nunique_colors++;
	  }  

	  /* replace color with index pointer in image structure */
	  /* (Read the image as color_values.  Write it as color_indices) */
	  color_indices[i] = ind; 
	}	  
	

	/* modify values in the Ximage structure to reflect new contents */
	image->bytes_per_line = image->bytes_per_line / 4; 
	image->bits_per_pixel = 8;
	image->depth          = 8;


	/* need to test the blue mask to see which endianness machine we are on. 
	   then grab the individual rgb values from the pixel value */
	/*	for (i=0; i<= nunique_colors; i++) { */

	if ( endian_type() == 65 ) {
	  if (ImageByteOrder(dpy))
	    reverse_bytes = True;
	  else
	    reverse_bytes = False;
	} else {
	  if (!ImageByteOrder(dpy))
	    reverse_bytes = True;
	  else
	    reverse_bytes = False;
	}
	/* only need to revers/swap pixel values if reverse_bytes is true */
	if (reverse_bytes) {
	  for (i=0; i< ncolors; i++) {
	    pixel = (*colors)[i].pixel;
	    cptr = (unsigned char *)&pixel;
	    tmp_cptr = cptr[0];
	    cptr[0]=cptr[3];
	    cptr[3]=tmp_cptr;
	    tmp_cptr = cptr[1];
	    cptr[1]=cptr[2];
	    cptr[2]=tmp_cptr;
	    (*colors)[i].pixel = pixel;	    
	  }
	}
	
	 XQueryColors(dpy, cmap, *colors, ncolors); 
	
    } else {
	for (i=0; i<ncolors; i++) {
	  (*colors)[i].pixel = i;
	  (*colors)[i].pad = 0;
	}
	XQueryColors(dpy, cmap, *colors, ncolors); 
    }

    /*    XQueryColors(dpy, cmap, *colors, ncolors);  */
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


int endian_type ()
{
  return (*(short *) "AZ")& 255;
}
