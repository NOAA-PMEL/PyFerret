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

#ifndef lint
    static char	rcsid[]	= "$Id$";
    static char	afsid[]	= "$__Header$";
#endif

#include "udposix.h"
#include <stdlib.h>
#include <stdio.h>
#include "xgks.h"


    static
WaitForBreak(wsid)
    Gint            wsid;
{
    Gchoice         init;
    Gchoicerec      record;
    Glimit          earea;

    earea.xmin = 0.0;
    earea.xmax = 1279.0;
    earea.ymin = 0.0;
    earea.ymax = 1023.0;

    init.status = GC_NOCHOICE;
    init.choice = 0;
    record.pet1.data = NULL;
    ginitchoice(wsid, 1, &init, 1, &earea, &record);
    gsetchoicemode(wsid, 1, GREQUEST, GECHO);
    for (; init.status != GC_NONE;)
	greqchoice(wsid, 1, &init);
}


main(argc, argv)
    /*ARGSUSED*/
    int             argc;
    char           *argv[];
{
    Gint            ws_id = 1;
    Gint            result;

    if ((result = gopengks(stdout, 0)) != 0)
	perr(result, "...open_gks");

    if ((result = gopenws(ws_id, argv[1], argv[1])) != 0)
	perr(result, "...open_ws");

    if ((result = gactivatews(ws_id)) != 0)
	perr(result, "...activate_ws");

    test_pline(ws_id);

    (void) fprintf(stderr, "Done, press break...\n");

    WaitForBreak(1);

    if ((result = gdeactivatews(ws_id)) != 0)
	perr(result, "...deactivate_ws");

    if ((result = gclosews(ws_id)) != 0)
	perr(result, "...close_ws");

    if ((result = gclosegks()) != 0)
	perr(result, "...close_gks");

    return 0;
}


perr(i, s)
    int             i;
    char           *s;
{
    (void) fprintf(stdout, "%s %d\n", s, i);
    exit(1);
}


#define BLACK	0
#define BLUE	1
#define GREEN	2
#define CYAN	3
#define RED	4
#define MAGENTA	5
#define YELLOW	6
#define WHITE	7


Gcobundl        Colors[] = {
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0},
    {0.0, 1.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 0.0, 0.0},
    {1.0, 0.0, 1.0},
    {1.0, 1.0, 0.0},
    {1.0, 1.0, 1.0}
};


LoadColors(wsid)
    Gint            wsid;
{
    int             i;

    for (i = BLACK; i <= WHITE; i++)
	gsetcolorrep(wsid, i, &Colors[i]);
}


int             lntbl[] = {GLN_LDASH, GLN_DDOTDASH, GLN_SDASH, GLN_SOLID,
			   GLN_DASH, GLN_DOT, GLN_DOTDASH};


test_pline(ws_id)
    Gint	    ws_id;
{
    Gpoint          lpts[2], tpt;
    Gint            i;
    char            s[20];
    Gpoint          up;
    Gtxfp           txfp;
    Gtxalign        align;

    LoadColors(ws_id);
    gsetdeferst(ws_id, GASAP, GALLOWED);

    txfp.font = 4;
    txfp.prec = GSTROKE;
    gsettextfontprec(&txfp);
    gsetcharexpan(0.5);
    gsetcharspace(0.2);
    gsettextcolorind(WHITE);

    gsetcharheight(0.05);
    up.x = 0.0;
    up.y = 1.0;
    gsetcharup(&up);
    align.hor = GTH_CENTER;
    align.ver = GTV_BASE;
    gsettextalign(&align);
    gsettextpath(GTP_RIGHT);

    tpt.x = 0.5;
    tpt.y = 0.9;
    gtext(&tpt, "GKS POLYLINES");

    txfp.font = 1;
    txfp.prec = GSTROKE;
    gsettextfontprec(&txfp);
    gsetcharheight(0.03);
    align.hor = GTH_RIGHT;
    align.ver = GTV_HALF;
    gsettextalign(&align);

    tpt.x = 0.15;
    tpt.y = 0.9;
    lpts[0].x = 0.2;
    lpts[0].y = 0.9;
    lpts[1].x = 0.9;
    lpts[1].y = 0.9;
    for (i = 1; i < 9; i++) {
	tpt.y -= 0.1;
	(void) sprintf(s, "width %5.2f", (float) (2.0 * i));
	gtext(&tpt, s);
	gsetlinetype(lntbl[(i % 7)]);
	gsetlinewidth((float) (2.0 * i));
	gsetlinecolorind((i % 7) + 1);
	lpts[0].y -= 0.1;
	lpts[1].y -= 0.1;
	gpolyline(2, lpts);
    }
}
