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
#   ifndef __SABER__
	static char	rcsid[]	= "$Id$";
	static char	afsid[]	= "$__Header$";
#   endif
#endif

/*
 * solve "tower of hanoi" problem for n disks
 * with graphical narration of solution
 */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <xgks.h>
#include "demo.h"

#define TOWERSPC 9.0
#define WINDWD 40.0
#define WINDHT WINDWD*0.75

/*
 * default values for command line options
 */
static int	n	= 5;


/*
 * Set up the color table.
 */
    static void
SetColor(id)
{
    Gcobundl	rep;

    rep.red	= 1.0;
    rep.green	= 1.0;
    rep.blue	= 1.0;
    gsetcolourrep(id, WHITE, &rep);

    rep.red	= 1.0;
    rep.green	= 0.0;
    rep.blue	= 0.0;
    gsetcolourrep(id, RED, &rep);

    rep.red	= 0.0;
    rep.green	= 1.0;
    rep.blue	= 0.0;
    gsetcolourrep(id, GREEN, &rep);

    rep.red	= 0.0;
    rep.green	= 0.0;
    rep.blue	= 1.0;
    gsetcolourrep(id, BLUE, &rep);

    rep.red	= 1.0;
    rep.green	= 1.0;
    rep.blue	= 0.0;
    gsetcolourrep(id, YELLOW, &rep);

    rep.red	= 0.0;
    rep.green	= 1.0;
    rep.blue	= 1.0;
    gsetcolourrep(id, CYAN, &rep);

    rep.red	= 1.0;
    rep.green	= 0.0;
    rep.blue	= 1.0;
    gsetcolourrep(id, VIOLET, &rep);

    rep.red	= 1.0;
    rep.green	= 0.5;
    rep.blue	= 0.0;
    gsetcolourrep(id, SILVER, &rep);

    rep.red	= 0.0;
    rep.green	= 1.0;
    rep.blue	= 0.5;
    gsetcolourrep(id, BEIGE, &rep);

    rep.red	= 1.0;
    rep.green	= 0.0;
    rep.blue	= 0.5;
    gsetcolourrep(id, DARKGREEN, &rep);

    return;
}


main(argc, argv)
    int             argc;
    char           *argv[];
{
    Gint            ws_id	= 1;
    Gint            mo_id	= 3;
    Glimit          WsWindow;
    char           *conn	= (char *) NULL;
    char           *mo_path	= "hanoi.gksm";
    int             c;
    extern char    *optarg;
    extern int      optind;

    WsWindow.xmin = 0.0;
    WsWindow.xmax = 1.0;
    WsWindow.ymin = 0.0;
    WsWindow.ymax = 0.8;

    while ((c = getopt(argc, argv, "d:n:o:")) != -1) {
	switch (c) {
	case 'd':
	    conn	= optarg;
	    break;
	case 'n':
	    n		= atoi(optarg);
	    break;
	case 'o':
	    mo_path	= optarg;
	    break;
	}
    }

    gopengks(stdout, 0);

    gopenws(ws_id, conn, conn);
    gopenws(mo_id, mo_path, "MO");

    gactivatews(ws_id);
    gactivatews(mo_id);

    SetColor(ws_id);
    SetColor(mo_id);

    gsetwswindow(ws_id, &WsWindow);
    gsetwswindow(mo_id, &WsWindow);

    title();

    /*
     * solve the problem
     */
    inittower(n);
    f(n, 0, 1, 2);

    /*
     * close workstation and GKS
     */
    WaitForBreak(ws_id);
    gdeactivatews(mo_id);
    gdeactivatews(ws_id);
    gclosews(mo_id);
    gclosews(ws_id);
    gclosegks();

    return 0;
}


/*
 * transfer n disks from tower a to tower b using c as a spare
 *
 *		transfer n-1 from a to c using b
 *		move 1 from a to b
 *		transfer n-1 from c to b using a
 */
f(n, a, b, c)
    int             n;
    int             a;
    int             b;
    int             c;
{
    if (n == 0)
	return;

    f(n - 1, a, c, b);
    movedisk(a, b);
    f(n - 1, c, b, a);
}


box(l)
    Glimit         *l;
{
    Gpoint          pts[5];

#define e 0.01

    pts[0].x = l->xmin + e;
    pts[0].y = l->ymin + e;
    pts[1].x = l->xmin + e;
    pts[1].y = l->ymax - e;
    pts[2].x = l->xmax - e;
    pts[2].y = l->ymax - e;
    pts[3].x = l->xmax - e;
    pts[3].y = l->ymin + e;
    pts[4].x = l->xmin + e;
    pts[4].y = l->ymin + e;
    gsetfillcolorind(WHITE);
    gfillarea(5, pts);
}


/*
 * print title across top of page
 */
title()
{
    Gpoint          p;
    Glimit          Window;
    Glimit          Viewport;
    Gtxfp           txfp;
    Gtxalign        txalign;

    Window.xmin = 0.0;
    Window.xmax = 16.0;
    Window.ymin = 0.0;
    Window.ymax = 2.0;

    Viewport.xmin = 0.1;
    Viewport.xmax = 0.9;
    Viewport.ymin = 0.58;
    Viewport.ymax = 0.74;

    txfp.font = 2;
    txfp.prec = GSTROKE;

    txalign.hor = GTH_CENTER;
    txalign.ver = GTV_HALF;

    gsetdeferst(1, GASAP, GALLOWED);
    gsetwindow(1, &Window);
    gsetviewport(1, &Viewport);
    gselntran(1);

    /*	box( &Window ); */

    txfp.font = 1;
    txfp.prec = GSTROKE;
    gsettextfontprec(&txfp);
    gsetcharspace(0.2);
    gsetcharheight(1.0);
    gsettextalign(&txalign);
    gsettextcolorind(YELLOW);

    p.x = 8.0;
    p.y = 1.0;
    gtext(&p, "Tower of Hanoi");
}


/*
 * initialize towers with all disks on tower 0.
 */
inittower(n)
    int             n;
{
    Glimit          Window;
    Glimit          Viewport;
    int             i;

    Window.xmin = 0.0;
    Window.xmax = WINDWD;
    Window.ymin = 0.0;
    Window.ymax = WINDHT;

    Viewport.xmin = 0.1;
    Viewport.xmax = 0.9;
    Viewport.ymin = 0.06;
    Viewport.ymax = 0.54;

    gsetwindow(1, &Window);
    gsetviewport(1, &Viewport);
    border(0.0, WINDWD, 0.0, WINDHT);

    for (i = n; i > 0; i--)
	placedisk(0, i);
}


/*
 * tower data structures and manipulation
 */

Gfloat          tcount[3] = {0.2, 0.2, 0.2};

int             towers[3][10] =
    {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

int             towerx[3] = {0, 0, 0};


movedisk(a, b)
{
    int             diskno;

    path(a, b);
    diskno = DiskRemove(a);
    unpath();
    placedisk(b, diskno);
}


placedisk(tower, diskno)
    int             tower, diskno;
{
    gsetfillcolorind(diskno);
    gsetfillintstyle(GSOLID);

    disk(diskno, TOWERSPC * (1 + tower) - (Gfloat) (diskno) / 
		 2.0, tcount[tower]);
    tcount[tower] += (Gfloat) diskno;
    pushdisk(tower, diskno);
}


DiskRemove(tower)
    int             tower;
{
    int             diskno;

    diskno = popdisk(tower);
    gsetfillcolorind(0);
    gsetfillintstyle(GSOLID);

    tcount[tower] -= (Gfloat) diskno;
    disk(diskno,
	 TOWERSPC * (1 + tower) - (Gfloat) (diskno) / 2.0, tcount[tower]);
    return diskno;
}


disk(diskno, x, y)
    int             diskno;
    Gfloat          x, y;
{
    Gpoint          pts[4];

#ifdef DEBUG
    printf("disk(%d, %8.3f, %8.3f\n", diskno, x, y);
#endif

    pts[0].x = x;
    pts[0].y = y;
    pts[1].x = x;
    pts[1].y = y + (Gfloat) diskno;
    pts[2].x = x + (Gfloat) diskno;
    pts[2].y = y + (Gfloat) diskno;
    pts[3].x = x + (Gfloat) diskno;
    pts[3].y = y;

    gfillarea(4, pts);
}


Gpoint          pathpts[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};


/*
 * draw a line from top disk of tower a to top of screen, then over
 * to tower b.
 */
path(a, b)
    int             a, b;
{
    gsetlinecolorind(GREEN);
    gsetlinetype(GLN_DASH);
    gsetlinewidth(3.0);
    pathpts[0].x = (a + 1) * TOWERSPC;
    pathpts[0].y = tcount[a] + (WINDHT * .05);
    pathpts[1].x = (a + 1) * TOWERSPC;
    pathpts[1].y = WINDHT - 1.0;
    pathpts[2].x = (b + 1) * TOWERSPC;
    pathpts[2].y = WINDHT - 1.0;
    pathpts[3].x = (b + 1) * TOWERSPC;
    pathpts[3].y = tcount[b] + (WINDHT * .05);

    gpolyline(4, pathpts);
}


unpath()
{
    gsetlinecolorind(0);
    gpolyline(4, pathpts);
}


border(x1, x2, y_1, y2)
    Gfloat          x1, x2, y_1, y2;
{
    Gpoint          pts[5];

    gsetlinecolorind(SILVER);
    gsetlinetype(GSOLID);
    gsetlinewidth(2.0);
    pts[0].x = x1;
    pts[0].y = y_1;
    pts[1].x = x1;
    pts[1].y = y2;
    pts[2].x = x2;
    pts[2].y = y2;
    pts[3].x = x2;
    pts[3].y = y_1;
    pts[4].x = x1;
    pts[4].y = y_1;

    gpolyline(5, pts);
}


pushdisk(tower, diskno)
    int             tower, diskno;
{
    towers[tower][towerx[tower]] = diskno;
    towerx[tower]++;
}


popdisk(tower)
    int             tower;
{

    towerx[tower]--;
    return towers[tower][towerx[tower]];
}
