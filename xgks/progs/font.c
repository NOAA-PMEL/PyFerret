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
    static char rcsid[] = "$Id$";
    static char afsid[] = "$__Header$";
#endif

#include "udposix.h"
#include <stdlib.h>		/* for atof() */
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "xgks.h"
#include "demo.h"

Gint            ws_id = 1;
Gint            result;
Glimit          w;
Glimit          v;
Glimit          wsw;
Glimit          wsv;


    static int
show_font(font)
    int             font;
{
    Gpoint          tpt;
    char            s[100];
    Gpoint          up;
    Gtxfp           txfp;
    Gtxalign        align;

    txfp.font = 1;
    txfp.prec = GSTROKE;
    if (gsettextfontprec(&txfp) != 0)
	return 0;

    /* gsetcharexpan(0.5); */
    /* gsetcharspace(0.2); */
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
    (void) sprintf(s, "GKS VECTOR FONT NUMBER %d", font);
    gtext(&tpt, s);

    gsetcharheight(0.05);
    align.hor = GTH_LEFT;
    align.ver = GTV_BASE;
    gsettextalign(&align);
    txfp.font = font;
    txfp.prec = GSTROKE;
    if (gsettextfontprec(&txfp) != 0)
	return 0;

    tpt.x = 0.01;
    tpt.y = 0.80;
    gtext(&tpt, " !\"#$%&'()*+,-./:;<=>?@");
    tpt.y -= 0.08;
    gtext(&tpt, "0123456789");
    tpt.y -= 0.08;
    gtext(&tpt, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    tpt.y -= 0.08;
    gtext(&tpt, "abcdefghijklmnopqrstuvwxyz");
    tpt.y -= 0.08;
    gtext(&tpt, "[\\]^_`{|}~");

    return 1;
}


main(argc, argv)
    int             argc;
    char           *argv[];
{
    int             i;
    int             c;
    char           *conn	= NULL;
    char           *mo_path	= "font.gksm";
    Gtxfac          fac;
    extern char    *optarg;
    extern int      optind;

    while ((c = getopt(argc, argv, "d:o:")) != -1) {
	switch (c) {
	case 'd':
	    conn	= optarg;
	    break;
	case 'o':
	    mo_path	= optarg;
	    break;
	}
    }

    gopengks(stdout, 0);

    if ((result = gopenws(ws_id, conn, conn)) != 0)
	perr(result, "...open_ws");
    gactivatews(ws_id);

    gopenws(200, mo_path, "MO");
    gactivatews(200);

    wsw.xmin = 0.0;
    wsw.xmax = 1.0;
    wsw.ymin = 0.0;
    wsw.ymax = 0.8;
    gsetwswindow(200, &wsw);

    wsv.xmin = 0.0;
    wsv.xmax = 1279.0;
    wsv.ymin = 0.0;
    wsv.ymax = 1023.0;
    gsetwsviewport(200, &wsv);

    w.xmin = 0.0;
    w.xmax = 1.0;
    w.ymin = 0.0;
    w.ymax = 1.0;
    gsetwindow(1, &w);

    v.xmin = 0.0;
    v.xmax = 1.0;
    v.ymin = 0.0;
    v.ymax = 0.8;
    gsetviewport(1, &v);

    gsetwswindow(ws_id, &wsw);
    gsetwsviewport(ws_id, &wsv);

    gselntran(1);

    ginqtextfacil((char*)NULL, &fac);
    (void) free((voidp)fac.fp_list);

    test_font(fac.fps);

    (void) fputs("Enter BREAK in window to continue\n", stderr);
    WaitForBreak(1);

    for (i = 1; i <= fac.fps; ++i) {

	gclearws(200, GALWAYS);
	gclearws(1, GALWAYS);

	if (!show_font(i))
	    break;

	(void) fputs("Enter BREAK in window to continue\n", stderr);
	WaitForBreak(1);
    }

    if ((result = gdeactivatews(200)) != 0)
	perr(result, "...deactivate_ws");

    if ((result = gclosews(200)) != 0)
	perr(result, "...close_ws");

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
    (void) fprintf(stdout, "%s %d\n", s, i);
    exit(1);
}


Gasfs           asf = {
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* polyline */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* polymarker */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL,	/* text */
    GINDIVIDUAL, GINDIVIDUAL, GINDIVIDUAL	/* fillarea */
};


test_font(nfonts)
    int             nfonts;
{
    Gpoint          tpt;
    char            s[100];
    int             i;
    Gpoint          up;
    Gtxfp           txfp;
    Gtxalign        align;

    txfp.font = 4;
    txfp.prec = GSTROKE;
    gsettextfontprec(&txfp);
    /* gsetcharexpan(0.5); */
    /* gsetcharspace(0.2); */
    gsettextcolorind(WHITE);

    gsetcharheight(0.05);
    up.x = 0.0;
    up.y = 1.0;
    gsetcharup(&up);
    align.hor = GTH_CENTER;
    align.ver = GTV_BASE;
    gsettextalign(&align);
    gsettextpath(GTP_RIGHT);

    gsetasf(&asf);

    tpt.x = 0.5;
    tpt.y = 0.9;
    gtext(&tpt, "GKS VECTOR FONTS");

    gsetcharheight(0.05);
    align.hor = GTH_LEFT;
    align.ver = GTV_BASE;
    gsettextalign(&align);

    tpt.x = 0.01;
    tpt.y = 0.90;
    for (i = 1; i <= nfonts; i++) {
	txfp.font = i;
	txfp.prec = GSTROKE;
	gsettextfontprec(&txfp);
	SetColor(i);
	tpt.y -= 0.08;
	(void) sprintf(s,
	    "Font #%d  AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz",
	    i);
	gtext(&tpt, s);
    }
}


SetColor(hash)
    Gint            hash;
{
    Gcobundl	rep;
    int		color_index	= hash%9;

    switch (color_index) {

    case 1:
	rep.red		= 1.0;
	rep.green	= 1.0;
	rep.blue	= 1.0;
	break;
    case 2:
	rep.red		= 1.0;
	rep.green	= 0.0;
	rep.blue	= 0.0;
	break;
    case 3:
	rep.red		= 0.0;
	rep.green	= 1.0;
	rep.blue	= 0.0;
	break;
    case 4:
	rep.red		= 0.0;
	rep.green	= 0.0;
	rep.blue	= 1.0;
	break;
    case 5:
	rep.red		= 1.0;
	rep.green	= 1.0;
	rep.blue	= 0.0;
	break;
    case 6:
	rep.red		= 0.0;
	rep.green	= 1.0;
	rep.blue	= 1.0;
	break;
    case 7:
	rep.red		= 1.0;
	rep.green	= 0.0;
	rep.blue	= 1.0;
	break;
    case 8:
	rep.red		= 1.0;
	rep.green	= 0.5;
	rep.blue	= 0.0;
	break;
    case 0:
	rep.red		= 0.0;
	rep.green	= 1.0;
	rep.blue	= 0.5;
	break;
    }

    gsetcolourrep(1, color_index, &rep);
    gsetcolourrep(200, color_index, &rep);
    gsettextcolorind(color_index);

    return;
}
