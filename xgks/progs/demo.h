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
 *
 * $Id$
 * $Header$
 */

#ifndef DEMO_H_INCLUDED
#define DEMO_H_INCLUDED

#define BLACK 0
#define WHITE 1
#define RED 2
#define GREEN 3
#define BLUE 4
#define YELLOW 5
#define CYAN 6
#define VIOLET 7
#define SILVER 8
#define BEIGE 9
#define DARKGREEN 10

#define DARKGRAY BEIGE
#define MEDIUMGRAY BEIGE

    static
WaitForBreak( ws_id )
    Gint       ws_id;
{
    Gchoice    init;
    Gchoicerec record;
    Glimit     earea;

    earea.xmin =    0.0;
    earea.xmax = 1279.0;
    earea.ymin =    0.0;
    earea.ymax = 1023.0;

    init.status      = GC_NOCHOICE;
    init.choice      = 0;
    record.pet1.data = NULL;
    ginitchoice( ws_id, 1, &init, 1, &earea, &record );
    gsetchoicemode( ws_id, 1, GREQUEST, GECHO );
    while (init.status != GC_NONE)
	greqchoice( ws_id, 1, &init );
}

#endif	/* DEMO_H_INCLUDED not defined */
