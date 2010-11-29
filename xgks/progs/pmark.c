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
#include <string.h>
#include "xgks.h"


    static
WaitForBreak(ws_id)
    Gint            ws_id;
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
    ginitchoice(ws_id, 1, &init, 1, &earea, &record);
    gsetchoicemode(ws_id, 1, GREQUEST, GECHO);
    for (; init.status != GC_NONE;)
	greqchoice(ws_id, 1, &init);
}


Gint            ws_id = 1;
Gint            result;


main(argc, argv)
    int             argc;
    char           *argv[];
{
    Gchar          *conn = (char *) NULL;
    Gint            i;

    for (i = 1; i < argc; i++) {
	if (strchr(argv[i], ':'))
	    conn = argv[i];
    }

    if ((result = gopengks(stdout, 0)) != 0)
	perr(result, "...open_gks");

    if ((result = gopenws(ws_id, conn, conn)) != 0)
	perr(result, "...open_ws");

    if ((result = gactivatews(ws_id)) != 0)
	perr(result, "...activate_ws");

    test_pmark();

    (void) fprintf(stderr, "Done, press break...\n");

    WaitForBreak(1);

    if ((result = gdeactivatews(ws_id)) != 0)
	perr(result, "...deactivate_ws");

    if ((result = gclosews(ws_id)) != 0)
	perr(result, "...close_ws");

    if ((result = gclosegks()) != 0)
	perr(result, "...close_gks");
    (void) fprintf(stdout, "after close_gks\n");

    return 0;
}


perr(i, s)
    int             i;
    char           *s;
{
    if (i)
	(void) fprintf(stdout, "%s %d\n", s, i);
    else
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


LoadColors(ws_id)
    Gint            ws_id;
{
    int             i;

    for (i = BLACK; i <= WHITE; i++)
	gsetcolorrep(ws_id, i, &Colors[i]);
}


Gasfs           IASFs = {
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* polyline */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* polymarker */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* text */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL	/* fillarea */
};

Gasfs           BASFs = {
    GBUNDLED, GBUNDLED, GBUNDLED,	/* polyline */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* polymarker */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* text */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL	/* fillarea */
};


test_pmark()
{
    Gpoint          tpt, pt;
    Gint            i, type;
    Gchar           s[20];
    Gpoint          up;
    Gtxfp           txfp;
    Gtxalign        align;

    LoadColors(ws_id);
    gsetdeferst(ws_id, GASAP, GALLOWED);

    txfp.font = 4;
    txfp.prec = GSTROKE;
    gsettextfontprec(&txfp);;
    gsetcharexpan(0.5);
    gsetcharspace(0.4);
    gsettextcolorind(WHITE);			/* WHITE */

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
    gtext(&tpt, "GKS POLYMARKERS");

    txfp.font = 1;
    txfp.prec = GSTROKE;
    gsettextfontprec(&txfp);
    gsetcharheight(0.03);
    align.hor = GTH_RIGHT;
    align.ver = GTV_HALF;
    gsettextalign(&align);

    tpt.x = 0.15;
    tpt.y = 0.9;
    pt.y = 0.9;
    for (i = 1; i < 9; i++) {
	tpt.y -= 0.1;
	(void) sprintf(s, "scale %5.2f", (float) (2.0 * i));
	gtext(&tpt, s);
	gsetmarkersize((float) (2.0 * i));
	gsetmarkercolorind((i % 7) + 1);
	pt.x = 0.25;
	pt.y -= 0.1;
	for (type = 1; type < 6; type++) {
	    gsetmarkertype(type);
	    gpolymarker(1, &pt);
	    pt.x += 0.16;
	}
    }
}
