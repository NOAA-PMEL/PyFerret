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
 */

/*
 * Metafiles
 */

#ifndef XGKS_CGM_H
#define XGKS_CGM_H


/*
 * CGM API:
 */
extern int CGMrecSize(
    Gint            type
);
extern int CGMnextItem(
    Metafile	*mf		/* Metafile structure */
);
extern int CGMwriteItem(
    Metafile	*mf,		/* Metafile structures */
    int		num,		/* Number of Metafiles */
    Gint	type,		/* item type */
    Gint	length,		/* item length */
    Gchar	*data		/* item data-record */
);
extern int CGMreadItem(
    Metafile	*mf,		/* Metafile structure  */
    char        *record	/* input data-record */
);
extern int CGMmiOpen(
    Metafile	*mf,		/* Metafile structure */
    char	*conn		/* Metafile identifier (filename) */
);
extern int CGMmiClose(
    Metafile	*mf		/* Metafile structure */
);
extern int CGMmoOpen(
    WS_STATE_PTR	ws
);
extern int CGMmoClose(
    Metafile	*mf
);
extern int CGMclear(
    Metafile	*mf,
    int		num,
    Gclrflag	flag
);
extern int CGMredrawAllSeg(
    Metafile	*mf,
    int		num
);
extern int CGMupdate(
    Metafile	*mf,
    int		num,
    Gregen	regenflag
);
extern int CGMdefer(
    Metafile	*mf,
    int		num,
    Gdefmode	defer_mode,
    Girgmode	regen_mode
);
extern int CGMmessage(
    Metafile	*mf,
    int		num,
    Gchar	*string
);
extern int CGMoutputGraphic(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	num_pt,
    Gpoint	*pos
);
extern int CGMtext(
    Metafile	*mf,
    int		num,
    Gpoint	*at,
    Gchar	*string
);
extern int CGMcellArray(
    Metafile	*mf,
    int		num,
    Gpoint	*ll,
    Gpoint	*ur,
    Gpoint	*lr,
    Gint	row,
    Gint	*colour,
    Gipoint	*dim
);
extern int CGMsetGraphSize(
    Metafile	*mf,
    int		num,
    Gint	code,
    double	size
);
extern int CGMcloseSeg(
    Metafile	*mf,
    int		num
);
extern int CGMsetGraphAttr(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	attr
);
extern int CGMsetTextFP(
    Metafile	*mf,
    int		num,
    Gtxfp	*txfp
);
extern int CGMsetCharUp(
    Metafile	*mf,
    int		num,
    Gpoint	*up,
    Gpoint	*base
);
extern int CGMsetTextPath(
    Metafile	*mf,
    int		num,
    Gtxpath	path
);
extern int CGMsetTextAlign(
    Metafile	*mf,
    int		num,
    Gtxalign	*align
);
extern int CGMsetFillStyle(
    Metafile	*mf,
    int		num,
    Gflinter	style
);
extern int CGMsetPatSize(
    Metafile	*mf,
    int		num
);
extern int CGMsetPatRefpt(
    Metafile	*mf,
    int		num
);
extern int CGMsetAsf(
    Metafile	*mf,
    int		num
);
extern int CGMsetLineMarkRep(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	idx,
    Gint	type,
    double	size,
    Gint	colour
);
extern int CGMsetTextRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gtxbundl	*rep
);
extern int CGMsetFillRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gflbundl	*rep
);
extern int CGMsetPatRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gptbundl	*rep
);
extern int CGMsetColRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gcobundl	*rep
);
extern int CGMsetClip(
    Metafile	*mf,
    int		num,
    Glimit	*rect
);
extern int CGMsetLimit(
    Metafile	*mf,
    int		num,
    Gint	code,
    Glimit	*rect
);
extern int CGMrenameSeg(
    Metafile	*mf,
    int		num,
    Gint	old,
    Gint	new
);
extern int CGMsetSegTran(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gfloat	matrix[2][3]
);
extern int CGMsetSegAttr(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gint	code,
    Gint	attr
);
extern int CGMsetSegVis(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gsegvis	vis
);
extern int CGMsetSegHilight(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gseghi	hilight
);
extern int CGMsetSegPri(
    Metafile	*mf,
    int		num,
    Gint	name,
    double	pri
);
extern int CGMsetSegDetect(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gsegdet	det
);

#endif	/* XGKS_CGM_H not defined above */
