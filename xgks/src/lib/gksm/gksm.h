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
 * This header-file depends upon header-file "xgks.h".
 * 
 * $Id$
 * $__Header$
 */

/*
 * Metafiles
 */

#ifndef XGKS_GKSM_H
#define XGKS_GKSM_H


extern int GMnextItem(
    Metafile	*mf		/* Metafile structure */
);
extern int GMwriteItem(
    Metafile	*mf,		/* Metafile structures */
    int		num,	/* Number of Metafiles */
    Gint	type,	/* item type */
    Gint	length,	/* item length */
    Gchar	*data	/* item data-record */
);
extern int GMreadItem(
    Metafile	*mf,		/* Metafile structure  */
    char        *record	/* input data-record */
);
extern int GMmiOpen(
    Metafile	*mf,		/* Metafile structure */
    char	*conn		/* Metafile identifier (filename) */
);
extern int GMmiClose(
    Metafile	*mf
);
extern int GMmoOpen(
    WS_STATE_PTR	ws
);
extern int GMmoClose(
    Metafile	*mf
);
extern int GMclear(
    Metafile	*mf,
    int		num,
    Gclrflag	flag
);
extern int GMredrawAllSeg(
    Metafile	*mf,
    int		num
);
extern int GMupdate(
    Metafile	*mf,
    int		num,
    Gregen	regenflag
);
extern int GMdefer(
    Metafile	*mf,
    int		num,
    Gdefmode	defer_mode,
    Girgmode	regen_mode
);
extern int GMmessage(
    Metafile	*mf,
    int		num,
    Gchar	*string
);
extern int GMoutputGraphic(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	num_pt,
    Gpoint	*pos
);
extern int GMtext(
    Metafile	*mf,
    int		num,
    Gpoint	*at,
    Gchar	*string
);
extern int GMcellArray(
    Metafile	*mf,
    int		num,
    Gpoint	*ll,
    Gpoint	*ur,
    Gpoint	*lr,
    Gint	row,
    Gint	*colour,
    Gipoint	*dim
);
extern int GMsetGraphSize(
    Metafile	*mf,
    int		num,
    Gint	code,
    double	size
);
extern int GMcloseSeg(
    Metafile	*mf,
    int		num
);
extern int GMsetGraphAttr(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	attr
);
extern int GMsetTextFP(
    Metafile	*mf,
    int		num,
    Gtxfp	*txfp
);
extern int GMsetCharUp(
    Metafile	*mf,
    int		num,
    Gpoint	*up,
    Gpoint	*base
);
extern int GMsetTextPath(
    Metafile	*mf,
    int		num,
    Gtxpath	path
);
extern int GMsetTextAlign(
    Metafile	*mf,
    int		num,
    Gtxalign	*align
);
extern int GMsetFillStyle(
    Metafile	*mf,
    int		num,
    Gflinter	style
);
extern int GMsetPatSize(
    Metafile	*mf,
    int		num
);
extern int GMsetPatRefpt(
    Metafile	*mf,
    int		num
);
extern int GMsetAsf(
    Metafile	*mf,
    int		num
);
extern int GMsetLineMarkRep(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	idx,
    Gint	type,
    double	size,
    Gint	colour
);
extern int GMsetTextRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gtxbundl	*rep
);
extern int GMsetFillRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gflbundl	*rep
);
extern int GMsetPatRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gptbundl	*rep
);
extern int GMsetColRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gcobundl	*rep
);
extern int GMsetClip(
    Metafile	*mf,
    int		num,
    Glimit	*rect
);
extern int GMsetLimit(
    Metafile	*mf,
    int		num,
    Gint	code,
    Glimit	*rect
);
extern int GMrenameSeg(
    Metafile	*mf,
    int		num,
    Gint	old,
    Gint	new
);
extern int GMsetSegTran(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gfloat	matrix[2][3]
);
extern int GMsetSegAttr(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gint	code,
    Gint	attr
);
extern int GMsetSegVis(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gsegvis	vis
);
extern int GMsetSegHilight(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gseghi	hilight
);
extern int GMsetSegPri(
    Metafile	*mf,
    int		num,
    Gint	name,
    double	pri
);
extern int GMsetSegDetect(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gsegdet	det
);

#endif	/* XGKS_GKSM_H not defined */
