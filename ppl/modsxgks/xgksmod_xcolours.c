/*
 *		Copyright IBM Corporation 1989
 *
 *                      All Rights Reserved
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appear in all copies and that
 * both that copyright notice and this permission notice appear in
 * supporting documentation, and that the name of IBM not be
 * used in advertising or publicity pertaining to distribution of the
 * software without specific, written prior permission.
 *
 * IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
 * ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
 * IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
 * ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
 * ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *
 *
 * University of Illinois at Urbana-Champaign
 * Department of Computer Science
 * 1304 W. Springfield Ave.
 * Urbana, IL	61801
 *
 * (C) Copyright 1987, 1988 by The University of Illinois Board of Trustees.
 * All rights reserved.
 *
 * Tool: X 11 Graphical Kernel System
 * Author: Gregory Scott Rogers
 * Author: Sung Hsien Ching Kelvin
 * Author: Yu Pan
 *
 * XGKS colour related functions
 *	xXgksSetColourRep():
 *	xXgksInqColourRep():
 */

/* Mod J Davison 7.31.95 to fix bug wherein XGKS attempts to free colors
 * it has not allocated.  This can be a problem when other applications are 
 * using private color maps and the pixel value given to free is in the 
 * private color map.  It happens because XcInit uses sequential values to 
 * initialize color indices without actually allocating the color.
 */

int XGKS_alloc_pixel[256]; /* Use to indicate pixel allocated by XGKS *jd* */

/*LINTLIBRARY*/

#include "udposix.h"
#include <stdlib.h>
#include <assert.h>
#include "gks_implem.h"

#ifdef lint
    static void	lint_malloc(n) size_t n; { n++; }
#   define	malloc(n)	(lint_malloc((n)), 0)
#else
    static char afsid[]	= "$__Header$";
    static char rcsid[]	= "$Id$";
#endif

#ifdef DEBUG
#   undef NDEBUG
#endif

/*
 * Macros for indexing into the color-conversion tables for visuals with
 * separate RGB palettes:
 */
#define IRED(map,color)    (((color) & (map)->red_mask)  /(map)->red_mult)
#define IGRN(map,color)    (((color) & (map)->green_mask)/(map)->green_mult)
#define IBLU(map,color)    (((color) & (map)->blue_mask) /(map)->blue_mult)

/*
 * Macros for computing the intensity bit-patterns for visuals with separate
 * RGB palettes:
 */
#define RED(map,ndx)	((ndx) * (map)->red_mult)
#define GRN(map,ndx)	((ndx) * (map)->green_mult)
#define BLU(map,ndx)	((ndx) * (map)->blue_mult)

/*
 * Macro for computing the color value (either GKS colour-index or X pixel-
 * value) for visuals with separate RGB palettes:
 */
#define RGB(map,tbl,color)	(\
	RED(map, (tbl)->rgb[IRED(map,color)].red) + \
	GRN(map, (tbl)->rgb[IGRN(map,color)].green) + \
	BLU(map, (tbl)->rgb[IBLU(map,color)].blue))

/*
 * Macro for computing the color value (either GKS colour-index or X pixel-
 * value) regardless of the visual class.
 */
#define COLOR(map,tbl,ndx)	\
	((map)->SeparateRGB ? RGB(map,tbl,ndx) : (tbl)->color[ndx])


extern int		XgksSIGIO_OFF();
extern int		XgksSIGIO_ON();

static unsigned long	MaskToMult();


    int
xXgksSetColourRep(ws, idx, rep)
    WS_STATE_PTR    ws;
    int             idx;
    Gcobundl       *rep;

{
    int             ncolours;

    if (ws->ewstype != X_WIN)
	return OK;

    /* restore the Signal Default Function */

    (void) XgksSIGIO_OFF(ws->dpy);

    /* initial some values and check the index value */

    ncolours = ws->wscolour;

    if (ncolours < 3) {
	(void) XgksSIGIO_ON(ws->dpy);		/* for black&white screen */
	return 0;
    }
    if ((idx < 0) || (idx >= ncolours)) {
	(void) XgksSIGIO_ON(ws->dpy);		/* index value out of the
	 * size of the colour map *//* c1147 d1 */
	return 1;
    }
    (void) XcSetColour(ws, idx, rep);		/* set color --SRE */

    /* Restore the interrupt of I/O signals */

    (void) XgksSIGIO_ON(ws->dpy);

    return 0;
}


    int
xXgksInqColourRep(ws, idx, type, rep)
    WS_STATE_PTR    ws;
    int             idx;

 /* ARGSUSED */
    Gqtype          type;
    Gcobundl       *rep;

{
    Display        *dpy;
    XColor          colour_ret;

    /*****************************************************************/
    /* NOTE: This routine is now only called for the GREALIZED case! */
    /*       When type == GSET, everything is handled in             */
    /*       ginqcolourrep() in colors.c  (DWO)                      */
    /*****************************************************************/

    if (ws->ewstype != X_WIN)
	return OK;

    /* restore the Signal Default Function */

    (void) XgksSIGIO_OFF(ws->dpy);

    /* check the validity of the index value */

    dpy = ws->dpy;

    /*
     * Removed check for valid idx here because that check has already been
     * done before this routine is called
     */

    /* get the RGB values */

    colour_ret.pixel = XcPixelValue(ws, idx);
    XQueryColor(dpy, ws->dclmp, &colour_ret);

    XAllocColor(dpy, ws->dclmp, &colour_ret);

    /* set the returned RGB values */

    rep->red = (Gfloat) colour_ret.red / 65535.0;
    rep->green = (Gfloat) colour_ret.green / 65535.0;
    rep->blue = (Gfloat) colour_ret.blue / 65535.0;

    (void) XgksSIGIO_ON(ws->dpy);

    return 0;
}


/*
 * WHAT:   Create a new instance of the color-index mapping abstraction.
 *
 * HOW:    Set the color-mapping tables in the GKS workstation-
 *         structure to NULL.
 *
 * INPUT:  Pointer to a GKS workstation-structure with assumed garbage
 *	   in the mapping-table member.
 *
 * OUTPUT: Success flag.
 */
XcNew(ws)
    WS_STATE_PTR    ws;			/* the GKS workstation */
{
    assert(ws != NULL);

    ws->XcMap.NumEntries	= 0;
    ws->XcMap.ToX.rgb		= NULL;
    ws->XcMap.ToGKS.rgb		= NULL;
    ws->XcMap.ToX.color		= NULL;
    ws->XcMap.ToGKS.color	= NULL;

    return 1;
}


/*
 * WHAT:   Initialize the color-mapping for the given display.
 *
 * HOW:    Allocate storage for the forward and inverse color-mapping tables
 *	   and set them to the identity transform (except for the GKS
 *	   foreground and background colors wich will have the following
 *	   mapping:
 *		GKS background <-> X WhitePixel()
 *		GKS foreground <-> X BlackPixel().
 *	   The size of the tables are based on the number of colors for the
 *	   workstation and the visual class.
 *
 * INPUT:  Pointer to a GKS workstation-structure with assumed garbage
 *	   in the color-mapping member and a visual information structure.
 *
 * OUTPUT: Success flag and modified GKS workstation-structure (color-mapping
 *	   member is initialized).
 */
XcInit(ws, vinfo)
    WS_STATE_PTR    ws;			/* the GKS workstation */
    XVisualInfo    *vinfo;		/* visual info for window */
{
    int             ReturnStatus = 0;	/* failure */
    XcMap          *map;
    XcTable        *ToX, *ToGKS;
    unsigned        nbytes;

    assert(ws != NULL);
    assert(ws->dpy != NULL);
    assert(vinfo != NULL);

    map = &ws->XcMap;
    ToX = &map->ToX;
    ToGKS = &map->ToGKS;

    map->NumEntries = vinfo->colormap_size;

    if (vinfo->class == TrueColor || vinfo->class == DirectColor) {
	nbytes = sizeof(XcRGB) * vinfo->colormap_size;

	map->SeparateRGB = 1;
	map->red_mask = vinfo->red_mask;
	map->green_mask = vinfo->green_mask;
	map->blue_mask = vinfo->blue_mask;
	map->red_mult = MaskToMult(vinfo->red_mask);
	map->green_mult = MaskToMult(vinfo->green_mask);
	map->blue_mult = MaskToMult(vinfo->blue_mask);

	if ((ToX->rgb = (XcRGB *) malloc((size_t)nbytes)) == NULL) {
	    (void) fprintf(stderr,
	       "XcInit: Couldn't allocate %u-bytes for GKS-to-X RGB-map.\n",
			   nbytes);
	} else {
	    if ((ToGKS->rgb = (XcRGB *) malloc((size_t)nbytes)) == NULL) {
		(void) fprintf(stderr,
		"XcInit: Couldn't allocate %u-bytes for X-to-GKS RGB-map.\n",
			       nbytes);
	    } else {
		register        i;

		/* Initialize mapping table with trivial mapping */
		for (i = 0; i < vinfo->colormap_size; ++i)
		    ToX->rgb[i].red
			= ToX->rgb[i].green
			= ToX->rgb[i].blue
			= ToGKS->rgb[i].red
			= ToGKS->rgb[i].green
			= ToGKS->rgb[i].blue
			= i;

		/* Set background GKS -> WhitePixel() */
		ToX->rgb[0].red = (unsigned) IRED(map,
			       WhitePixel(ws->dpy, DefaultScreen(ws->dpy)));
		ToX->rgb[0].green = (unsigned) IGRN(map,
			       WhitePixel(ws->dpy, DefaultScreen(ws->dpy)));
		ToX->rgb[0].blue = (unsigned) IBLU(map,
			       WhitePixel(ws->dpy, DefaultScreen(ws->dpy)));

		/* Set foreground GKS -> BlackPixel() */
		ToX->rgb[1].red = (unsigned) IRED(map,
			       BlackPixel(ws->dpy, DefaultScreen(ws->dpy)));
		ToX->rgb[1].green = (unsigned) IGRN(map,
			       BlackPixel(ws->dpy, DefaultScreen(ws->dpy)));
		ToX->rgb[1].blue = (unsigned) IBLU(map,
			       BlackPixel(ws->dpy, DefaultScreen(ws->dpy)));

		/* Set WhitePixel() -> GKS background */
		ToGKS->rgb[ToX->rgb[0].red].red = 0;
		ToGKS->rgb[ToX->rgb[0].green].green = 0;
		ToGKS->rgb[ToX->rgb[0].blue].blue = 0;

		/* Set BlackPixel() -> GKS foreground */
		ToGKS->rgb[ToX->rgb[1].red].red = ToX->rgb[1].red;
		ToGKS->rgb[ToX->rgb[1].green].green = ToX->rgb[1].green;
		ToGKS->rgb[ToX->rgb[1].blue].blue = ToX->rgb[1].blue;

		ReturnStatus = 1;
	    }
	}
    } else {					/* single palette */
	map->SeparateRGB = 0;

	nbytes = sizeof(unsigned long) * vinfo->colormap_size;

	if ((ToX->color = (unsigned long *) malloc((size_t)nbytes)) == NULL) {
	    (void) fprintf(stderr,
	     "XcInit: Couldn't allocate %u-bytes for GKS-to-X color-map.\n",
			   nbytes);
	} else {
	    if ((ToGKS->color = (unsigned long *) malloc((size_t)nbytes)) 
		    == NULL) {
		(void) fprintf(stderr,
	   "XcInit: Couldn't allocate %u-bytes for X-to-GKS color-map.\n",
			       nbytes);
	    } else {
		register        i;

		/* Initialize mapping table with trivial mapping */
		for (i = 0; i < vinfo->colormap_size; ++i)
		  {
		    ToX->color[i] = ToGKS->color[i] = i;
		    XGKS_alloc_pixel[i] = 0; /* Haven't actually allocated */
		  }

		/* Set background GKS -> WhitePixel() */
		ToX->color[0] = WhitePixel(ws->dpy, DefaultScreen(ws->dpy));

		/* Set foreground GKS -> BlackPixel() */
		ToX->color[1] = BlackPixel(ws->dpy, DefaultScreen(ws->dpy));
	
		/* Set WhitePixel() -> background GKS */
		ToGKS->color[ToX->color[0]] = 0;
		XGKS_alloc_pixel[ToX->color[0]] = 1; /* jd */

		/* Set BlackPixel() -> foreground GKS */
		ToGKS->color[ToX->color[1]] = 1;
		XGKS_alloc_pixel[ToX->color[1]] = 1; /* jd */

		ReturnStatus = 1;
	    }
	}
    }

    return ReturnStatus;
}


/*
 * Compute the color-index multiplier corresponding to a color-mask.
 * See chapter 7 (Color) in the Xlib Programming Manual for a discussion
 * of these concepts.
 */
    static unsigned long
MaskToMult(mask)
    unsigned long   mask;
{
    unsigned long   mult;

    for (mult = 1; mult != 0; mult <<= 1)
	if (mask & mult)
	    break;

    return mult;
}


/*
 * WHAT:   Set the color associated with a GKS color-index.
 *
 * HOW:	   Get the X-server color in the default X colormap that is closest
 *	   to the desired GKS color and store it in the mapping tables.
 *
 * INPUT:  Pointer to a GKS workstation-structure; a GKS color-index;
 *	   and a GKS representation of the desired color.
 *
 * OUTPUT: Success flag (0 => failure) and modified color-index mapping-table.
 */
XcSetColour(ws, GKSindex, GKSrep)
    WS_STATE_PTR    ws;			/* the GKS workstation */
    Gint            GKSindex;		/* GKS color-index */
    Gcobundl       *GKSrep;		/* GKS color-representation */
{
    int             ReturnStatus = 0;	/* failure */
    XColor          Xrep;		/* X color-representation */

    assert(ws != NULL);
    assert(GKSindex >= 0);
    assert(GKSrep != NULL);

    /* Convert GKS [0.-1.] representation to X (unsigned short) rep. */
    Xrep.red = 65535 * GKSrep->red;
    Xrep.green = 65535 * GKSrep->green;
    Xrep.blue = 65535 * GKSrep->blue;

    /*
     * Get the X-server color closest to the desired GKS one and save its
     * color-cell index (i.e. pixel-value) in the table.  Also, set the
     * inverse (i.e. X-to-GKS) color-transformation
     */
    if (XAllocColor(ws->dpy, ws->dclmp, &Xrep)) {
	XcMap          *map = &ws->XcMap;
	XcTable        *ToX = &map->ToX;
	XcTable        *ToGKS = &map->ToGKS;

	/* Pixel successfully allocated by XGKS */
	XGKS_alloc_pixel[Xrep.pixel] = 1; /* jd */

	if (map->SeparateRGB) {
	    ToX->rgb[IRED(map, GKSindex)].red
		= (unsigned) IRED(map, Xrep.pixel);
	    ToX->rgb[IGRN(map, GKSindex)].green
		= (unsigned) IGRN(map, Xrep.pixel);
	    ToX->rgb[IBLU(map, GKSindex)].blue
		= (unsigned) IBLU(map, Xrep.pixel);

	    ToGKS->rgb[IRED(map, Xrep.pixel)].red
		= (unsigned) IRED(map, GKSindex);
	    ToGKS->rgb[IGRN(map, Xrep.pixel)].green
		= (unsigned) IGRN(map, GKSindex);
	    ToGKS->rgb[IBLU(map, Xrep.pixel)].blue
		= (unsigned) IBLU(map, GKSindex);
	} else {
	    ToX->color[GKSindex] = Xrep.pixel;
	    ToGKS->color[Xrep.pixel] = (unsigned long) GKSindex;
	}

	ReturnStatus = 1;

    } else {
	static int	SecondTry	= 0;	/* second attempt? */

	if (SecondTry) {
	    (void) fprintf(stderr,
    "XcSetColour: Couldn't allocate X color: RGB = %u %u %u.\n",
			   Xrep.red, Xrep.green, Xrep.blue);
	} else {
	    unsigned long	pixel	= XcPixelValue(ws, GKSindex);
	    unsigned long	planes	= 0;

	    /* Return if pixel not allocated *jd* */
	    if (!XGKS_alloc_pixel[pixel]) {
	      (void) fprintf(stderr,
		   "XcSetColour: Couldn't allocate X color: RGB = %u %u %u.\n",
			   Xrep.red, Xrep.green, Xrep.blue);
	      return ReturnStatus;
	    }

	    if (XFreeColors(ws->dpy, ws->dclmp, &pixel, 1, planes) == 0) {
	        XGKS_alloc_pixel[pixel] = 0; /* Now a pixel unallocated *jd* */
		SecondTry	= 1;
		ReturnStatus	= XcSetColour(ws, GKSindex, GKSrep);
		SecondTry	= 0;
	    }
	}
    }

    return ReturnStatus;
}


/*
 * WHAT:   Map a GKS color-index to an X pixel-value (i.e. color-cell
 *	   index).
 *
 * HOW:	   Use the GKS-to-X color-mapping to determine the pixel-value
 *	   -- either by simple lookup (for non true-color visuals) or
 *	   by computation (for true-color visuals).
 *
 * INPUT:  Pointer to a GKS workstation-structure WITH A VALID DISPLAY (i.e.
 *	   with valid "dpy").
 *
 * OUTPUT: X pixel-value corresponding to GKS color-index.  Out-of-range
 *	   GKS color-indices are mapped to the nearest X pixel-value.
 */
    unsigned long
XcPixelValue(ws, ColourIndex)
    WS_STATE_PTR    ws;			/* the GKS workstation */
    Gint            ColourIndex;	/* GKS color-index */
{
    XcMap          *map;		/* color mapping */
    unsigned long   PixelValue;		/* returned value */

    assert(ws != NULL);
    assert(ColourIndex >= 0);

    map = &ws->XcMap;

    assert(map != NULL);

    if (ColourIndex < 0) {
	ColourIndex = 0;
    } else if (ColourIndex >= map->NumEntries) {
	ColourIndex = map->NumEntries - 1;
    }
    PixelValue = (unsigned long) COLOR(map, &map->ToX, ColourIndex);

    return PixelValue;
}


/*
 * WHAT:   Map an X pixel-value (i.e. color-cell index) to an GKS colour-index.
 *
 * HOW:	   Use the X-to-GKS color-mapping to determine the pixel-value
 *	   -- either by simple lookup (for non true-color visuals) or
 *	   by computation (for true-color visuals).
 *
 * INPUT:  Pointer to a GKS workstation-structure WITH A VALID DISPLAY (i.e.
 *	   with valid "dpy").
 *
 * OUTPUT: GKS colour-index corresponding to X pixel-value.  Out-of-range
 *	   X pixel-values are mapped to the nearest GKS colour-index.
 */
    Gint
XcColourIndex(ws, PixelValue)
    WS_STATE_PTR    ws;			/* the GKS workstation */
    unsigned long   PixelValue;		/* X pixel-value */
{
    XcMap          *map;		/* color mapping */
    Gint            ColourIndex;	/* returned value */

    assert(ws != NULL);

    map = &ws->XcMap;

    assert(map != NULL);

    if (PixelValue >= map->NumEntries)
	PixelValue = map->NumEntries - 1;

    ColourIndex = (unsigned long) COLOR(map, &map->ToGKS, PixelValue);

    return ColourIndex;
}


/*
 * WHAT:   Terminate use of the mapping-table in the given workstation
 *	   structure.
 *
 * HOW:	   Free-up allocated storage, if necessary, and set the mapping-tables
 *	   to NULL pointers.
 *
 * INPUT:  Pointer to a GKS workstation-structure WITH A NON-GARBAGE "XcTable"
 *	   MEMBER.
 *
 * OUTPUT: Success flag and modified workstation structure with NULL mapping-
 *	   tables.
 */
XcEnd(ws)
    WS_STATE_PTR    ws;			/* the GKS workstation */
{
    XcMap          *map;
    XcTable        *ToX, *ToGKS;

    assert(ws != NULL);

    map = &ws->XcMap;
    ToX = &map->ToX;
    ToGKS = &map->ToGKS;

    if (map->SeparateRGB) {
	if (ToX->rgb != NULL) {
	    ufree((voidp)ToX->rgb);
	    ToX->rgb = NULL;
	}
	if (ToGKS->rgb != NULL) {
	    ufree((voidp) ToGKS->rgb);
	    ToGKS->rgb = NULL;
	}
    } else {
	if (ToX->color != NULL) {
	    ufree((voidp) ToX->color);
	    ToX->color = NULL;
	}
	if (ToGKS->color != NULL) {
	    ufree((voidp)ToGKS->color);
	    ToGKS->color = NULL;
	}
    }

    return 1;
}

/*
 * find the number of colour table entries supportted by an X server.
 * returns the number of entries or -1 if the server does not respond.
 *
 * Moved from XGKS colours.c to use correct visual when determining number
 * of available colors
 */
XgksMaxColours(server)
    char           *server;
{
    int             i, colours;
    Display        *dpy;
    char           *getenv();


    /* wait till dpy is known to turn SIGIO off  AIX PORT #d1 */

    /* default server is in the Unix environment */
    if (server == NULL)
	server = getenv("DISPLAY");

    /* check for existing connection to this server. */
    for (i = 0; i < MAX_OPEN_WS; i++) {
	if (xgks_state.openedws[i].ws_id == INVALID
		|| xgks_state.openedws[i].ws->ewstype != X_WIN)
	    continue;
	if (STRCMP(xgks_state.openedws[i].ws->wstype, server) == 0)
	    break;
    }
    if (i < MAX_OPEN_WS) {			/* found a connection */
	dpy = xgks_state.openedws[i].ws->dpy;
	(void) XgksSIGIO_OFF(dpy);
	colours = xgks_state.openedws[i].ws->wscolour;
    } else {					/* build a connection */
	dpy = XOpenDisplay(server);
	(void) XgksSIGIO_OFF(dpy);
	if (dpy == NULL)
	    return -1;
	colours = DisplayCells(dpy, DefaultScreen(dpy));
	XCloseDisplay(dpy);
    }

    (void) XgksSIGIO_ON(dpy);

    return colours;
}
