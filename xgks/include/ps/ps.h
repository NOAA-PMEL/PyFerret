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
 * PostScript driver for XGKS metafiles
 * Created by Joe Sirott, Pacific Marine Environmental Lab
 * $Id$
 */


#ifndef XGKS_PS_H
#define XGKS_PS_H


extern int PSrecSize(
    Gint            type
);
extern int PSnextItem(
    Metafile	*mf		/* Metafile structure */
);
extern int PSwriteItem(
    Metafile      **mf,		/* Metafile structures */
    int             num,	/* Number of Metafiles */
    Gint	    type,	/* item type */
    Gint	    length,	/* item length */
    Gchar          *data	/* item data-record */
);
extern int PSreadItem(
    Metafile	*mf,		/* Metafile structure  */
    char        *record	/* input data-record */
);
extern int PSmiOpen(
    Metafile	*mf		/* Metafile structure */
);
extern int PSmoOpen(
    WS_STATE_PTR ws
);
extern int PSmoClose(
    Metafile	*mf
);
extern int PSclear(
    Metafile	*mf,
    int		num,
    Gclrflag	flag
);
extern int PSredrawAllSeg(
    Metafile	**mf,
    int		num
);
extern int PSupdate(
    Metafile	**mf,
    int		num,
    Gregen	regenflag
);
extern int PSdefer(
    Metafile	**mf,
    int		num,
    Gdefmode	defer_mode,
    Girgmode	regen_mode
);
extern int PSmessage(
    Metafile	**mf,
    int		num,
    Gchar	*string
);
extern int PSoutputGraphic(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	num_pt,
    Gpoint	*pos
);
extern int PStext(
    Metafile	*mf,
    int		num,
    Gpoint	*at,
    Gchar	*string
);
extern int PScellArray(
    Metafile	*mf,
    int		num,
    Gpoint	*ll,
    Gpoint	*ur,
    Gpoint	*lr,
    Gint	row,
    Gint	*colour,
    Gipoint	*dim
);
extern int PSsetGraphSize(
    Metafile	*mf,
    int		num,
    Gint	code,
    double	size
);
extern int PScloseSeg(
    Metafile	*mf,
    int		num
);
extern int PSsetGraphAttr(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	attr
);
extern int PSsetTextFP(
    Metafile	*mf,
    int		num,
    Gtxfp	*txfp
);
extern int PSsetCharUp(
    Metafile	*mf,
    int		num,
    Gpoint	*up,
    Gpoint	*base
);
extern int PSsetTextPath(
    Metafile	*mf,
    int		num,
    Gtxpath	path
);
extern int PSsetTextAlign(
    Metafile	*mf,
    int		num,
    Gtxalign	*align
);
extern int PSsetFillStyle(
    Metafile	*mf,
    int		num,
    Gflinter	style
);
extern int PSsetPatSize(
    Metafile	*mf,
    int		num
);
extern int PSsetPatRefpt(
    Metafile	*mf,
    int		num
);
extern int PSsetAsf(
    Metafile	*mf,
    int		num
);
extern int PSsetLineMarkRep(
    Metafile	*mf,
    int		num,
    Gint	code,
    Gint	idx,
    Gint	type,
    double	size,
    Gint	colour
);
extern int PSsetTextRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gtxbundl	*rep
);
extern int PSsetFillRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gflbundl	*rep
);
extern int PSsetPatRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gptbundl	*rep
);
extern int PSsetColRep(
    Metafile	*mf,
    int		num,
    Gint	idx,
    Gcobundl	*rep
);
extern int PSsetClip(
    Metafile	*mf,
    int		num,
    Glimit	*rect
);
extern int PSsetLimit(
    Metafile	*mf,
    int		num,
    Gint	code,
    Glimit	*rect
);
extern int PSrenameSeg(
    Metafile	*mf,
    int		num,
    Gint	old,
    Gint	new
);
extern int PSsetSegTran(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gfloat	matrix[2][3]
);
extern int PSsetSegAttr(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gint	code,
    Gint	attr
);
extern int PSsetSegVis(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gsegvis	vis
);
extern int PSsetSegHilight(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gseghi	hilight
);
extern int PSsetSegPri(
    Metafile	*mf,
    int		num,
    Gint	name,
    double	pri
);
extern int PSsetSegDetect(
    Metafile	*mf,
    int		num,
    Gint	name,
    Gsegdet	det
);

#endif	/* XGKS_PS_H not defined */
