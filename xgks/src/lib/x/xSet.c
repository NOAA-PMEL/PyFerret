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
 * University of Illinois at Urbana-Champaign
 * Department of Computer Science
 * 1304 W. Springfield Ave.
 * Urbana, IL	61801
 *
 * (C) Copyright 1987, 1988 by The University of Illinois Board of Trustees.
 *	All rights reserved.
 *
 * Tool: X 11 Graphical Kernel System
 * Author: Gregory Scott Rogers
 * Author: Sung Hsien Ching Kelvin
 * Author: Yu Pan
 */

/*LINTLIBRARY*/

#include "udposix.h"
#include <stdlib.h>
#include "gks_implem.h"

#ifndef lint
    static char afsid[]	= "$__Header$";
    static char rcsid[]	= "$Id$";
#endif


xXgksSetForeground(dpy, gc, fg)
    Display        *dpy;
    GC              gc;
    unsigned long   fg;
{
    XSetForeground(dpy, gc, fg);
}


xXgksSetLineAttributes(dpy, gc, line_width, line_style, cap_style, join_style)
    Display        *dpy;
    GC              gc;
    unsigned int    line_width;
    int             line_style, cap_style, join_style;
{

    XSetLineAttributes(dpy, gc, line_width, line_style, cap_style,
		       join_style);
}


xXgksSetStipple(dpy, gc, stipple)
    Display        *dpy;
    GC              gc;
    Pixmap          stipple;
{
    XSetStipple(dpy, gc, stipple);
}


xXgksSetDashes(dpy, gc, ws, i)
    Display        *dpy;
    GC              gc;
    WS_STATE_PTR    ws;
    Gint            i;
{
    if (i != ws->last_dash_index)
	XSetDashes(dpy, gc, 0, xgksDASHES[i].dashl, xgksDASHES[i].dn);
}



xXgksSetTile(dpy, gc, tile)
    Display        *dpy;
    GC              gc;
    Pixmap          tile;
{
    XSetTile(dpy, gc, tile);
}


xXgksSetClipMask(dpy, gc, pixmap)
    Display        *dpy;
    GC              gc;
    Pixmap          pixmap;
{
    XSetClipMask(dpy, gc, pixmap);
}


xXgksSetPlineClipRectangles(dpy, gc, ws, rectangle)
    Display        *dpy;
    GC              gc;
    WS_STATE_PTR    ws;
    XRectangle     *rectangle;
{
    if ((rectangle->x != ws->last_pline_rectangle.x)
	    || (rectangle->y != ws->last_pline_rectangle.y)
	    || (rectangle->width != ws->last_pline_rectangle.width)
	    || (rectangle->height != ws->last_pline_rectangle.height)) {
	XSetClipRectangles(dpy, gc, 0, 0, rectangle, 1, Unsorted);
	ws->last_pline_rectangle = *rectangle;
    }
}


xXgksSetPmarkerClipRectangles(dpy, gc, ws, rectangle)
    Display        *dpy;
    GC              gc;
    WS_STATE_PTR    ws;
    XRectangle     *rectangle;
{
    if ((rectangle->x != ws->last_pmarker_rectangle.x)
	    || (rectangle->y != ws->last_pmarker_rectangle.y)
	    || (rectangle->width != ws->last_pmarker_rectangle.width)
	    || (rectangle->height != ws->last_pmarker_rectangle.height)) {
	XSetClipRectangles(dpy, gc, 0, 0, rectangle, 1, Unsorted);
	ws->last_pmarker_rectangle = *rectangle;
    }
}


xXgksSetFillAreaClipRectangles(dpy, gc, ws, rectangle)
    Display        *dpy;
    GC              gc;
    WS_STATE_PTR    ws;
    XRectangle     *rectangle;
{
    if ((rectangle->x != ws->last_farea_rectangle.x)
	    || (rectangle->y != ws->last_farea_rectangle.y)
	    || (rectangle->width != ws->last_farea_rectangle.width)
	    || (rectangle->height != ws->last_farea_rectangle.height)) {
	XSetClipRectangles(dpy, gc, 0, 0, rectangle, 1, Unsorted);
	ws->last_farea_rectangle = *rectangle;
    }
}


xXgksSetTextClipRectangles(dpy, gc, ws, rectangle)
    Display        *dpy;
    GC              gc;
    WS_STATE_PTR    ws;
    XRectangle     *rectangle;
{
    if ((rectangle->x != ws->last_text_rectangle.x)
	    || (rectangle->y != ws->last_text_rectangle.y)
	    || (rectangle->width != ws->last_text_rectangle.width)
	    || (rectangle->height != ws->last_text_rectangle.height)) {
	XSetClipRectangles(dpy, gc, 0, 0, rectangle, 1, Unsorted);
	ws->last_text_rectangle = *rectangle;
    }
}


xXgksSetFillStyle(dpy, gc, fill_style)
    Display        *dpy;
    GC              gc;
    int             fill_style;
{
    XSetFillStyle(dpy, gc, fill_style);
}
