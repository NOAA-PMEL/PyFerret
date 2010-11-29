/*
 * This file contains the interface to the metafile implementations of XGKS.
 * It is mostly a "switching" interface in that it does little work itself, 
 * but rather calls a specific metafile implementation that does the actual 
 * work.  Currently, two types of metafile implementations are available: 
 * GKSM and CGM.  For historical reasons, the default is GKSM.
 *
 * Copyright (C) 1991 UCAR/Unidata
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose without fee is hereby granted, provided
 * that the above copyright notice appear in all copies, that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of UCAR/Unidata not be used in
 * advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.  UCAR makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.  It is
 * provided with no support and without obligation on the part of UCAR or
 * Unidata, to assist in its use, correction, modification, or enhancement.
 *
 */


/*
 * Mod *jd* 6.15.94 to fix bug in number of metafile ws colors available.
 * It was 16, now is 256. 
 */

/* Fixed bug below for linux port (received 'warning') *jd* 1.28.97 
 * define MO_SET_SEG_DETECTmf, num, name, det)	<-- original
 * define MO_SET_SEG_DETECT(mf, num, name, det)	<-- fix
 */

/*
 * Modified by Joe Sirott, Pacific Marine Environmental Lab
 * Added support for PostScript and GIF metafile output.
 * Also, changed prototypes from K&R to ANSI C.
 *
 * all macros that call SEL_FUNC must call it with metafile *, not
 *   metafile**    *js* 8.97
 */

#define MAX_META_WSCOLOURS 256

/*LINTLIBRARY*/

#include <wchar.h>
#include "udposix.h"
#include <stdlib.h>
#include <time.h>		/* for time(), localtime(), and strftime() */
#include <sys/types.h>		/* for uid_t */
#include <unistd.h>		/* for getuid() & getlogin() */
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>
#include "gks_implem.h"
#include "gksm/gksm.h"
#include "cgm/cgm.h"
#include "ps/ps.h"
#include "gif/gif.h"

#ifndef lint
    static char rcsid[]	= "$Id$";
    static char afsid[]	= "$__Header$";
#endif

#define NOTSUPPORTED(type)	(type==6 || type==16)

/*
 * List of active output Metafiles:
 */
static int	num_cgmo;
static int	num_gksmo;
static int      num_psmo;
static int      num_gifmo;
static Metafile	active_cgmo[MAX_ACTIVE_WS];
static Metafile	active_gksmo[MAX_ACTIVE_WS];
static Metafile	active_psmo[MAX_ACTIVE_WS];
static Metafile	active_gifmo[MAX_ACTIVE_WS];

typedef int (*INT_PROC)();

/*
 * Keep in old K&R C to avoid painfu, picky function prototyping requirements
 * of MIPS c compiler
 */
static INT_PROC SEL_FUNC(mf, gksm, cgm, ps, gif)
     Metafile *mf; INT_PROC gksm; INT_PROC cgm; INT_PROC ps; INT_PROC gif;
{
    switch(mf->any->type){
      case MF_GKSM:
	return gksm;
      case MF_PS:
	return ps;
      case MF_CGM:
	return cgm;
      case MF_GIF:
	return gif;
      default:
	return 0;
    }
}

static INT_PROC emptyProc(void)
{
    return (INT_PROC)0;
}

/*
 * The following are macros that expand to the appropriate type of metafile
 * function (i.e. either GKSM or CGM or PostScript).
 *
 * This allows the metafile functions to be chosen at the time of metafile 
 * creation
 */

#define MO_CELL_ARRAY(mf, num, ll, ur, lr, row, colour, dim) \
	    SEL_FUNC(mf, (INT_PROC)GMcellArray, (INT_PROC)CGMcellArray, \
		     (INT_PROC)PScellArray,  (INT_PROC)GIFcellArray)\
	    (mf, num, ll, ur, lr, row, colour, dim)
#define MO_CLEAR(mf, num, flag)	\
	    SEL_FUNC(mf, (INT_PROC)GMclear, (INT_PROC)CGMclear, (INT_PROC)PSclear, (INT_PROC)GIFclear)(mf, num, flag)
#define MO_CLOSE(mf)	\
	    SEL_FUNC(mf, (INT_PROC)GMmoClose, (INT_PROC)CGMmoClose, (INT_PROC)PSmoClose, (INT_PROC)GIFmoClose)(mf)
#define MO_CLOSE_SEG(mf, num)	\
	    SEL_FUNC(mf, (INT_PROC)GMcloseSeg, (INT_PROC)CGMcloseSeg, (INT_PROC)PScloseSeg, (INT_PROC)GIFcloseSeg)(mf, num)
#define MO_DEFER(mf, num, defer_mode, regen_mode)	\
	    SEL_FUNC(mf, (INT_PROC)GMdefer, (INT_PROC)CGMdefer, (INT_PROC)PSdefer, (INT_PROC)GIFdefer)(mf, num, defer_mode, regen_mode)
#define MI_GET_NEXT_ITEM(mf)	\
	    SEL_FUNC(mf, (INT_PROC)GMnextItem, (INT_PROC)CGMnextItem, emptyProc, emptyProc)(mf)
#define MO_MESSAGE(mf, num, string)	\
	    SEL_FUNC(mf, (INT_PROC)GMmessage, (INT_PROC)CGMmessage, (INT_PROC)PSmessage, (INT_PROC)GIFmessage)(mf, num, string)
#define MI_OPEN(mf)	\
	    SEL_FUNC(mf, (INT_PROC)GMmiOpen, (INT_PROC)CGMmiOpen, emptyProc, emptyProc)
#define MO_OPEN(mf)	\
	    SEL_FUNC(mf, (INT_PROC)GMmoOpen, (INT_PROC)CGMmoOpen, (INT_PROC)PSmoOpen, (INT_PROC)GIFmoOpen)(mf)
#define MI_CLOSE(mf)	\
	    SEL_FUNC(mf, GMmiClose, CGMmiClose(mf), emptyProc, emptyProc)(mf)
#define MI_READ_ITEM(mf, record)	\
	    SEL_FUNC(mf, (INT_PROC)GMreadItem, (INT_PROC)CGMreadItem, emptyProc, emptyProc)(mf, record)
#define MO_REDRAW_ALL_SEG(mf, num)	\
	    SEL_FUNC(mf, (INT_PROC)GMredrawAllSeg, (INT_PROC)CGMredrawAllSeg, (INT_PROC)PSredrawAllSeg, (INT_PROC)GIFredrawAllSeg)(mf, num)
#define MO_RENAME_SEG(mf, num, old, new)	\
	    SEL_FUNC(mf, (INT_PROC)GMrenameSeg, (INT_PROC)CGMrenameSeg, (INT_PROC)PSrenameSeg, (INT_PROC)GIFrenameSeg)(mf, num, old, new)
#define MO_SET_ASF(mf, num)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetAsf, (INT_PROC)CGMsetAsf, (INT_PROC)PSsetAsf, (INT_PROC)GIFsetAsf)(mf, num)
#define MO_SET_CHAR_UP(mf, num, up, base)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetCharUp, (INT_PROC)CGMsetCharUp, (INT_PROC)PSsetCharUp, (INT_PROC)GIFsetCharUp)(mf, num, up, base)
#define MO_SET_CLIPPING(mf, num, rect)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetClip, (INT_PROC)CGMsetClip, (INT_PROC)PSsetClip, (INT_PROC)GIFsetClip)(mf, num, rect)
#define MO_SET_COLOUR_REP(mf, num, idx, rep)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetColRep, (INT_PROC)CGMsetColRep, (INT_PROC)PSsetColRep, (INT_PROC)GIFsetColRep)(mf, num, idx, rep)
#define MO_SET_FILL_REP(mf, num, idx, rep)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetFillRep, (INT_PROC)CGMsetFillRep, (INT_PROC)PSsetFillRep, (INT_PROC)GIFsetFillRep)(mf, num, idx, rep)
#define MO_SET_FILL_STYLE(mf, num, style)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetFillStyle, (INT_PROC)CGMsetFillStyle, (INT_PROC)PSsetFillStyle, (INT_PROC)GIFsetFillStyle)(mf, num, style)
#define MO_SET_GRAPH_SIZE(mf, num, code, size)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetGraphSize, (INT_PROC)CGMsetGraphSize, (INT_PROC)PSsetGraphSize, (INT_PROC)GIFsetGraphSize)(mf, num, code, size)
#define MO_SET_GRAPH_ATTR(mf, num, code, attr)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetGraphAttr, (INT_PROC)CGMsetGraphAttr, (INT_PROC)PSsetGraphAttr, (INT_PROC)GIFsetGraphAttr)(mf, num, code, attr)
#define MO_SET_LIMIT(mf, num, code, rect)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetLimit, (INT_PROC)CGMsetLimit, (INT_PROC)PSsetLimit, (INT_PROC)GIFsetLimit)(mf, num, code, rect)
#define MO_SET_LINE_MARKER_REP(mf, num, code, idx, type, size, colour)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetLineMarkRep, (INT_PROC)CGMsetLineMarkRep, (INT_PROC)PSsetLineMarkRep, (INT_PROC)GIFsetLineMarkRep)\
		(mf, num, code, idx, type, size, colour)
#define MO_SET_PATTERN_REFPT(mf, num)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetPatRefpt, (INT_PROC)CGMsetPatRefpt, (INT_PROC)PSsetPatRefpt, (INT_PROC)GIFsetPatRefpt)(mf, num)
#define MO_SET_PATTERN_REP(mf, num, idx, rep)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetPatRep, (INT_PROC)CGMsetPatRep, (INT_PROC)PSsetPatRep, (INT_PROC)GIFsetPatRep)(mf, num, idx, rep)
#define MO_SET_PATTERN_SIZE(mf, num)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetPatSize, (INT_PROC)CGMsetPatSize, (INT_PROC)PSsetPatSize, (INT_PROC)GIFsetPatSize)(mf, num)
#define MO_SET_SEG_ATTR(mf, num, name, code, attr)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetSegAttr, (INT_PROC)CGMsetSegAttr, (INT_PROC)PSsetSegAttr, (INT_PROC)GIFsetSegAttr)(mf, num, name, code, attr)
#define MO_SET_SEG_DETECT(mf, num, name, det)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetSegDetect, (INT_PROC)CGMsetSegDetect, (INT_PROC)PSsetSegDetect, (INT_PROC)GIFsetSegDetect)(mf, num, name, det)
#define MO_SET_SEG_HILIGHT(mf, num, name, hilight)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetSegHilight, (INT_PROC)CGMsetSegHilight, (INT_PROC)PSsetSegHilight, (INT_PROC)GIFsetSegHilight\
		(mf, num, name, hilight))
#define MO_SET_SEG_PRI(mf, num, name, pri)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetSegPri, (INT_PROC)CGMsetSegPri, (INT_PROC)PSsetSegPri, (INT_PROC)GIFsetSegPri)(mf, num, name, pri)
#define MO_SET_SEG_TRANS(mf, num, name, matrix)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetSegTran, (INT_PROC)CGMsetSegTran, (INT_PROC)PSsetSegTran, (INT_PROC)GIFsetSegTran)(mf, num, name, matrix)
#define MO_SET_SEG_VIS(mf, num, name, vis)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetSegVis, (INT_PROC)CGMsetSegVis, (INT_PROC)PSsetSegVis, (INT_PROC)GIFsetSegVis)(mf, num, name, vis)
#define MO_SET_TEXT_ALIGN(mf, num, align)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetTextAlign, (INT_PROC)CGMsetTextAlign, (INT_PROC)PSsetTextAlign, (INT_PROC)GIFsetTextAlign)(mf, num, align)
#define MO_SET_TEXT_FP(mf, num, txfp)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetTextFP, (INT_PROC)CGMsetTextFP, (INT_PROC)PSsetTextFP, (INT_PROC)GIFsetTextFP)(mf, num, txfp)
#define MO_SET_TEXT_PATH(mf, num, path)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetTextPath, (INT_PROC)CGMsetTextPath, (INT_PROC)PSsetTextPath, (INT_PROC)GIFsetTextPath)(mf, num, path)
#define MO_SET_TEXT_REP(mf, num, idx, rep)	\
	    SEL_FUNC(mf, (INT_PROC)GMsetTextRep, (INT_PROC)CGMsetTextRep, (INT_PROC)PSsetTextRep, (INT_PROC)GIFsetTextRep)(mf, num, idx, rep)
#define MO_UPDATE(mf, num, regenflag)	\
	    SEL_FUNC(mf, (INT_PROC)GMupdate, (INT_PROC)CGMupdate, (INT_PROC)PSupdate, (INT_PROC)GIFupdate)(mf, num, regenflag)
#define MO_GRAPHIC(mf, num, code, num_pt, pos)	\
	    SEL_FUNC(mf, (INT_PROC)GMoutputGraphic, (INT_PROC)CGMoutputGraphic, (INT_PROC)PSoutputGraphic, (INT_PROC)GIFoutputGraphic)\
		(mf, num, code, num_pt, pos)
#define MO_WRITE_ITEM(mf, num, type, length, data)	\
	    SEL_FUNC(mf, (INT_PROC)GMwriteItem, (INT_PROC)CGMwriteItem, emptyProc, emptyProc)(mf, num, type, length, data)
#define MO_TEXT(mf, num, at, string)	\
	    SEL_FUNC(mf, (INT_PROC)GMtext, (INT_PROC)CGMtext, (INT_PROC)PStext, (INT_PROC)GIFtext)(mf, num, at, string)


/*
 * Execute the data contained in a Metafile item.
 */
    static Gint
XgksExecData(Gint type, char *record)
{
    XGKSMONE       *ptr1;
    XGKSMTWO       *ptr2;
    XGKSMMESG      *msg;
    XGKSMGRAPH     *graph;
    XGKSMTEXT      *text;
    XGKSMSIZE      *size;
    XGKSMCHARVEC   *vec;
    XGKSMASF       *asf;
    XGKSMLMREP     *lmrep;
    XGKSMTEXTREP   *txrep;
    XGKSMFILLREP   *flrep;
    XGKSMPATREP    *patrep;
    XGKSMCOLOURREP *corep;
    XGKSMLIMIT     *limit;
    XGKSMSEGTRAN   *tran;
    XGKSMSEGPRI    *pri;
    XGKSMCELLARRAY *cell;
    XGKSMPATREF    *patref;
    XGKSMPATSIZ    *patsiz;
    OUT_PRIMI      *primi;
    Gtxpath         path;
    Gflinter        inter;
    Gdefmode        defmode;
    Gtxfp           txfp;
    Gsegattr        segattr;
    Gtxalign        txalign;
    Glnbundl        lnrep;
    Gmkbundl        mkrep;
    Gtxbundl        textrep;
    Gflbundl        fillrep;
    Gptbundl        ptrep;
    Gcobundl        colourrep;
    Gasfs           asfs;
    Gint            cnt, i, j;
    Gpoint         *pts;
    Gpoint          siz;
    Gfloat          height;

    switch (type) {

    case 0:
	break;

    case 2:
	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gredrawsegws(xgks_state.activews[cnt].ws_id);
		}
	    }
	}
	break;
    case 82:
	/*
	 * only need to call gcloseseg() once, not for each workstation
	 */
	(void) gcloseseg();
	break;

    case 1:
	ptr1 = (XGKSMONE *) record;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		(void) gclearws(xgks_state.activews[cnt].ws_id, 
				(ptr1->flag == 0 ? GCONDITIONALLY : GALWAYS));
	    }
	}
	break;
    case 3:
	ptr1 = (XGKSMONE *) record;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gupdatews(xgks_state.activews[cnt].ws_id,
				     (ptr1->flag == 0 ? GPERFORM : GPOSTPONE));
		}
	    }
	}
	break;
    case 21:
	ptr1 = (XGKSMONE *) record;
	(void) gsetlineind(ptr1->flag);
	break;
    case 22:
	ptr1 = (XGKSMONE *) record;
	(void) gsetlinetype(ptr1->flag);
	break;
    case 24:
	ptr1 = (XGKSMONE *) record;
	(void) gsetlinecolourind(ptr1->flag);
	break;
    case 25:
	ptr1 = (XGKSMONE *) record;
	(void) gsetmarkerind(ptr1->flag);
	break;
    case 26:
	ptr1 = (XGKSMONE *) record;
	(void) gsetmarkertype(ptr1->flag);
	break;
    case 28:
	ptr1 = (XGKSMONE *) record;
	(void) gsetmarkercolourind(ptr1->flag);
	break;
    case 29:
	ptr1 = (XGKSMONE *) record;
	(void) gsettextind(ptr1->flag);
	break;
    case 33:
	ptr1 = (XGKSMONE *) record;
	(void) gsettextcolourind(ptr1->flag);
	break;
    case 35:
	ptr1 = (XGKSMONE *) record;
	if (ptr1->flag == 0)
	    path = GTP_RIGHT;
	else if (ptr1->flag == 1)
	    path = GTP_LEFT;
	else if (ptr1->flag == 2)
	    path = GTP_UP;
	else
	    path = GTP_DOWN;
	(void) gsettextpath(path);
	break;
    case 37:
	ptr1 = (XGKSMONE *) record;
	(void) gsetfillind(ptr1->flag);
	break;
    case 38:
	ptr1 = (XGKSMONE *) record;
	if (ptr1->flag == 0)
	    inter = GHOLLOW;
	else if (ptr1->flag == 1)
	    inter = GSOLID;
	else if (ptr1->flag == 2)
	    inter = GPATTERN;
	else
	    inter = GHATCH;
	(void) gsetfillintstyle(inter);
	break;
    case 39:
	ptr1 = (XGKSMONE *) record;
	(void) gsetfillstyleind(ptr1->flag);
	break;
    case 40:
	ptr1 = (XGKSMONE *) record;
	(void) gsetfillcolourind(ptr1->flag);
	break;
    case 41:
	patsiz = (XGKSMPATSIZ *) record;
	siz.x = patsiz->wid.x;
	siz.y = patsiz->hgt.y;
	(void) gsetpatsize(&siz);
	break;
    case 42:
	patref = (XGKSMPATREF *) record;
	(void) gsetpatrefpt(&patref->ref);
	break;
    case 44:
	ptr1 = (XGKSMONE *) record;
	(void) gsetpickid(ptr1->flag);
	break;
    case 81:
	ptr1 = (XGKSMONE *) record;
	(void) gcreateseg(ptr1->flag);
	break;
    case 84:
	ptr1 = (XGKSMONE *) record;
	(void) gdelseg(ptr1->flag);
	break;

    case 4:
	ptr2 = (XGKSMTWO *) record;
	if (ptr2->item1 == 0)
	    defmode = GASAP;
	else if (ptr2->item1 == 1)
	    defmode = GBNIG;
	else if (ptr2->item1 == 2)
	    defmode = GBNIL;
	else
	    defmode = GASTI;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetdeferst(xgks_state.activews[cnt].ws_id,
				       defmode,
				       (ptr2->item2 == 0 
					   ? GALLOWED 
					   : GSUPPRESSED));
		}
	    }
	}
	break;
    case 30:
	ptr2 = (XGKSMTWO *) record;
	txfp.font = ptr2->item1;
	if (ptr2->item2 == 0)
	    txfp.prec = GSTRING;
	else if (ptr2->item2 == 1)
	    txfp.prec = GCHAR;
	else
	    txfp.prec = GSTROKE;
	(void) gsettextfontprec(&txfp);
	break;
    case 36:
	ptr2 = (XGKSMTWO *) record;
	if (ptr2->item1 == 0)
	    txalign.hor = GTH_NORMAL;
	else if (ptr2->item1 == 1)
	    txalign.hor = GTH_LEFT;
	else if (ptr2->item1 == 2)
	    txalign.hor = GTH_CENTRE;
	else
	    txalign.hor = GTH_RIGHT;
	if (ptr2->item2 == 0)
	    txalign.ver = GTV_NORMAL;
	else if (ptr2->item2 == 1)
	    txalign.ver = GTV_TOP;
	else if (ptr2->item2 == 2)
	    txalign.ver = GTV_CAP;
	else if (ptr2->item2 == 3)
	    txalign.ver = GTV_HALF;
	else if (ptr2->item2 == 4)
	    txalign.ver = GTV_BASE;
	else
	    txalign.ver = GTV_BOTTOM;
	(void) gsettextalign(&txalign);
	break;
    case 83:
	ptr2 = (XGKSMTWO *) record;
	(void) grenameseg(ptr2->item1, ptr2->item2);
	break;
    case 92:
	ptr2 = (XGKSMTWO *) record;
	segattr.seg = ptr2->item1;
	(void) ginqsegattr(&segattr);
	segattr.vis = (ptr2->item2 == 0 ? GVISIBLE : GINVISIBLE);
	(void) gsetsegattr(ptr2->item1, &segattr);
	break;
    case 93:
	ptr2 = (XGKSMTWO *) record;
	segattr.seg = ptr2->item1;
	(void) ginqsegattr(&segattr);
	segattr.hilight = (ptr2->item2 == 0 ? GNORMAL : GHIGHLIGHTED);
	(void) gsetsegattr(ptr2->item1, &segattr);
	break;
    case 95:
	ptr2 = (XGKSMTWO *) record;
	segattr.seg = ptr2->item1;
	(void) ginqsegattr(&segattr);
	segattr.det = (ptr2->item2 == 0 ? GUNDETECTABLE : GDETECTABLE);
	(void) gsetsegattr(ptr2->item1, &segattr);
	break;

    case 5:
	msg = (XGKSMMESG *) record;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gmessage(xgks_state.activews[cnt].ws_id,
				    msg->string);
		}
	    }
	}
	break;

    case 11:
	graph = (XGKSMGRAPH *) record;
	GKSERROR(((pts = (Gpoint*)malloc((size_t) (graph->num_pts*
			sizeof(Gpoint)))) == NULL),
		 300,
		 errXgksExecData);
	for (i = 0; i < graph->num_pts; i++)
	    NdcToWc(&(graph->pts[i]), &(pts[i]));
	(void) gpolyline(graph->num_pts, pts);
	ufree((voidp)pts);
	break;
    case 12:
	graph = (XGKSMGRAPH *) record;
	GKSERROR(((pts = (Gpoint *) malloc((size_t) (graph->num_pts *
			sizeof(Gpoint)))) == NULL),
		 300,
		 errXgksExecData);
	for (i = 0; i < graph->num_pts; i++)
	    NdcToWc(&(graph->pts[i]), &(pts[i]));
	(void) gpolymarker(graph->num_pts, pts);
	ufree((voidp)pts);
	break;
    case 14:
	graph = (XGKSMGRAPH *) record;
	GKSERROR(((pts = (Gpoint *) malloc((size_t) (graph->num_pts *
			sizeof(Gpoint)))) == NULL),
		 300,
		 errXgksExecData);
	for (i = 0; i < graph->num_pts; i++)
	    NdcToWc(&(graph->pts[i]), &(pts[i]));
	(void) gfillarea(graph->num_pts, pts);
	ufree((voidp)pts);
	break;

    case 13:
	text = (XGKSMTEXT *) record;
	GKSERROR(((pts = (Gpoint *) malloc(sizeof(Gpoint))) == NULL),
		 300, errXgksExecData);
	NdcToWc(&(text->location), pts);
	(void) gtext(pts, text->string);
	ufree((voidp)pts);
	break;

    case 15:
	cell = (XGKSMCELLARRAY *) record;
	GKSERROR(((primi = XgksNewPrimi()) == NULL), 300, errXgksExecData);
	primi->pid = CELL_ARRAY;
	primi->primi.cell_array.dim = cell->dim;
	/* rowsize is equal to cell->dim.x */
	primi->primi.cell_array.rowsize = cell->dim.x;
	j = cell->dim.x * cell->dim.y;
	GKSERROR(((primi->primi.cell_array.colour = (Gint *) 
			malloc((size_t) (j * sizeof(Gint)))) == NULL),
		 300,
		 errXgksExecData);
	primi->primi.cell_array.ll = cell->ll;
	primi->primi.cell_array.ur = cell->ur;
	primi->primi.cell_array.lr = cell->lr;
	primi->primi.cell_array.ul.x = cell->ll.x + (cell->ur.x - cell->lr.x);
	primi->primi.cell_array.ul.y = cell->ll.y + (cell->ur.y - cell->lr.y);
	for (i = 0; i < j; i++)
	    primi->primi.cell_array.colour[i] = cell->colour[i];
	XgksProcessPrimi(primi);
	if (MO_OPENED == TRUE)
	    XgksMoCellArray(&(primi->primi.cell_array.ll),
			    &(primi->primi.cell_array.ur),
			    &(primi->primi.cell_array.lr),
			    primi->primi.cell_array.rowsize,
			    primi->primi.cell_array.colour,
			    &(primi->primi.cell_array.dim));
	ufree((voidp)primi->primi.cell_array.colour);
	ufree((voidp)primi);
	break;

    case 23:
	size = (XGKSMSIZE *) record;
	(void) gsetlinewidth(size->size);
	break;
    case 27:
	size = (XGKSMSIZE *) record;
	(void) gsetmarkersize(size->size);
	break;
    case 31:
	size = (XGKSMSIZE *) record;
	(void) gsetcharexpan(size->size);
	break;
    case 32:
	size = (XGKSMSIZE *) record;
	(void) gsetcharspace(size->size);
	break;

    case 34:
	vec = (XGKSMCHARVEC *) record;
	VecNdcToWc(&(vec->up), &siz);
	height = sqrt((siz.x * siz.x) + (siz.y * siz.y));
	xgks_state.gks_chattr.up.x = siz.x / height;
	xgks_state.gks_chattr.up.y = siz.y / height;
	xgks_state.gks_chattr.height = height;
	VecNdcToWc(&(vec->base), &siz);
	height = sqrt((siz.x * siz.x) + (siz.y * siz.y));
	xgks_state.gks_chattr.base.x = siz.x / height;
	xgks_state.gks_chattr.base.y = siz.y / height;
	xgks_state.gks_chattr.chwidth = height;
	break;

    case 43:
	asf = (XGKSMASF *) record;
	asfs.ln_type = (asf->asf[0] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.ln_width = (asf->asf[1] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.ln_colour = (asf->asf[2] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.mk_type = (asf->asf[3] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.mk_size = (asf->asf[4] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.mk_colour = (asf->asf[5] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.tx_fp = (asf->asf[6] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.tx_exp = (asf->asf[7] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.tx_space = (asf->asf[8] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.tx_colour = (asf->asf[9] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.fl_inter = (asf->asf[10] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.fl_style = (asf->asf[11] == 0 ? GBUNDLED : GINDIVIDUAL);
	asfs.fl_colour = (asf->asf[12] == 0 ? GBUNDLED : GINDIVIDUAL);
	(void) gsetasf(&asfs);
	break;

    case 51:
	lmrep = (XGKSMLMREP *) record;
	lnrep.type = lmrep->style;
	lnrep.width = lmrep->size;
	lnrep.colour = lmrep->colour;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetlinerep(xgks_state.activews[cnt].ws_id,
				       lmrep->idx, &lnrep);
		}
	    }
	}
	break;
    case 52:
	lmrep = (XGKSMLMREP *) record;
	mkrep.type = lmrep->style;
	mkrep.size = lmrep->size;
	mkrep.colour = lmrep->colour;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetmarkerrep(xgks_state.activews[cnt].ws_id,
					 lmrep->idx, &mkrep);
		}
	    }
	}
	break;

    case 53:
	txrep = (XGKSMTEXTREP *) record;
	textrep.fp.font = txrep->font;
	textrep.ch_exp = txrep->tx_exp;
	textrep.space = txrep->space;
	textrep.colour = txrep->colour;
	if (txrep->prec == 0)
	    textrep.fp.prec = GSTRING;
	else if (txrep->prec == 1)
	    textrep.fp.prec = GCHAR;
	else
	    textrep.fp.prec = GSTROKE;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsettextrep(xgks_state.activews[cnt].ws_id,
				       txrep->idx, &textrep);
		}
	    }
	}
	break;

    case 54:
	flrep = (XGKSMFILLREP *) record;
	fillrep.style = flrep->style;
	fillrep.colour = flrep->colour;
	if (flrep->intstyle == 0)
	    fillrep.inter = GHOLLOW;
	else if (flrep->intstyle == 1)
	    fillrep.inter = GSOLID;
	else if (flrep->intstyle == 2)
	    fillrep.inter = GPATTERN;
	else
	    fillrep.inter = GHATCH;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetfillrep(xgks_state.activews[cnt].ws_id,
				       flrep->idx, &fillrep);
		}
	    }
	}
	break;

    case 55:
	patrep = (XGKSMPATREP *) record;
	ptrep.size.x = patrep->size.x;
	ptrep.size.y = patrep->size.y;
	j = ptrep.size.x * ptrep.size.y;
	GKSERROR(((ptrep.array = (Gint *) malloc((size_t) (j *
			sizeof(Gint)))) == NULL),
		 300,
		 errXgksExecData);
	for (i = 0; i < j; i++)
	    ptrep.array[i] = patrep->array[i];

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {

		/* don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetpatrep(xgks_state.activews[cnt].ws_id,
				      patrep->idx, &ptrep);
		}
	    }
	}
	break;

    case 56:
	corep = (XGKSMCOLOURREP *) record;
	colourrep.red = corep->red;
	colourrep.green = corep->green;
	colourrep.blue = corep->blue;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {

		/* don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetcolourrep(xgks_state.activews[cnt].ws_id,
					 corep->idx, &colourrep);
		}
	    }
	}
	break;

    case 61:
	limit = (XGKSMLIMIT *) record;
	xgks_state.cliprec.rec = limit->rect;
	XgksProcessClip(&xgks_state.cliprec.rec);
	break;

    case 71:
	limit = (XGKSMLIMIT *) record;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {

		/* don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetwswindow(xgks_state.activews[cnt].ws_id,
					&(limit->rect));
		}
	    }
	}
	break;
    case 72:
	limit = (XGKSMLIMIT *) record;

	/* Do it on all active ws */
	for (cnt = 0; cnt < MAX_ACTIVE_WS; cnt++) {
	    if (xgks_state.activews[cnt].ws_id != INVALID) {
		/* c1138:  don't do this on WISS */
		if (xgks_state.activews[cnt].ws->ewstype != WISS) {
		    (void) gsetwsviewport(xgks_state.activews[cnt].ws_id,
					  &(limit->rect));
		}
	    }
	}
	break;

    case 91:
	tran = (XGKSMSEGTRAN *) record;
	segattr.seg = tran->name;
	(void) ginqsegattr(&segattr);
	segattr.segtran[0][0] = tran->matrix[0][0];
	segattr.segtran[0][1] = tran->matrix[0][1];
	segattr.segtran[0][2] = tran->matrix[0][2];
	segattr.segtran[1][0] = tran->matrix[1][0];
	segattr.segtran[1][1] = tran->matrix[1][1];
	segattr.segtran[1][2] = tran->matrix[1][2];
	(void) gsetsegattr(tran->name, &segattr);
	break;

    case 94:
	pri = (XGKSMSEGPRI *) record;
	segattr.seg = pri->name;
	(void) ginqsegattr(&segattr);
	segattr.pri = pri->pri;
	(void) gsetsegattr(pri->name, &segattr);
	break;

    default:
	return 1;
    }
    return 0;
}


/*
 * Indicate whether or not a Metafile item of the given type is valid.
 */
    static int
XgksValidGksMItem(Gint type)
{
    if (type >= 0 && type <= 6)
	return OK;
    if (type >= 11 && type <= 16)
	return OK;
    if (type >= 21 && type <= 44)
	return OK;
    if (type >= 51 && type <= 56)
	return OK;
    if (type == 61)
	return OK;
    if (type == 71 || type == 72)
	return OK;
    if (type >= 81 && type <= 84)
	return OK;
    if (type >= 91 && type <= 95)
	return OK;
    if (type > 100)
	return OK;

    return INVALID;
}


/*
 * WRITE ITEM TO GKSM
 */
    int
gwritegksm(Gint ws_id, Gint type, Gint length, Gchar *data)
                          	/* workstation identifier */
        	         	/* item type */
        	           	/* item length */
                         	/* item data-record */
{
    WS_STATE_PTR    ws;
    Metafile       *mf;

    GKSERROR((xgks_state.gks_state != GWSAC && xgks_state.gks_state != GSGOP), 
	     5, errgwritegksm);

    GKSERROR((!VALID_WSID(ws_id)), 20, errgwritegksm);

    /* if it isn't open, it can't be active... */
    GKSERROR(((ws = OPEN_WSID(ws_id)) == NULL), 30, errgwritegksm);

    GKSERROR((ws->wsstate != GACTIVE), 30, errgwritegksm);

    GKSERROR((WS_CAT(ws) != GMO), 32, errgwritegksm);

    GKSERROR((type <= 100), 160, errgwritegksm);

    GKSERROR((length < 0), 161, errgwritegksm);

    mf	= &ws->mf;

    return MO_WRITE_ITEM(mf, 1, type, length, data);
}


/*
 * GET ITEM TYPE FROM GKSM
 */
    int
ggetgksm(Gint ws_id, Ggksmit *result)
                          	/* workstation identifier */
                           	/* input metafile item information */
{
    WS_STATE_PTR    ws;

    GKSERROR((xgks_state.gks_state != GWSOP && xgks_state.gks_state != 
		GWSAC && xgks_state.gks_state != GSGOP),
	     7,
	     errggetgksm);

    /* check for invalid workstation id */
    GKSERROR((!VALID_WSID(ws_id)), 20, errggetgksm);

    GKSERROR(((ws = OPEN_WSID(ws_id)) == NULL), 25, errggetgksm);

    GKSERROR((WS_CAT(ws) != GMI), 34, errggetgksm);

    GKSERROR((ws->mf.any->GksmEmpty == TRUE), 162, errggetgksm);

    if (XgksValidGksMItem(ws->mf.any->CurItem.type) == INVALID)
	ws->mf.any->filestat = MF_ITEM_ERR;
    GKSERROR((ws->mf.any->filestat != METAFILE_OK), 163, errggetgksm);

    *result	= ws->mf.any->CurItem;

    return OK;
}


/*
 * READ ITEM FROM GKSM
 *
 * The filestat field has been added to the workstation state structure to
 * retain MI error information between calls to ggetgksm and greadgksm.  The
 * field has one of four possible integer values (defined in metafile.h):
 *    METAFILE_OK -- no errors so far reading from metafile
 *    MF_DATA_ERR -- type and length of latest item read (current item) are
 *                   ok, but XgksReadData couldn't read the data (eg. non-
 *                   numeric characters found when trying to read an integer)
 *                   The item may be skipped (via greadgksm w/ length 0) and
 *                   MI processing can continue.
 *    MF_ITEM_ERR -- something more serious than a data error found in latest
 *                   item; eg. type invalid, length invalid, data read ter-
 *                   minated prematurely.  This error condition can be detected
 *                   while going on to the next item, so the current item is
 *                   returned correctly, but subsequent attempts to get/read
 *                   will fail.  Since the exact cause of the error is unknown,
 *                   this is not a recoverable condition.
 *    MF_FILE_ERR -- the system reported an I/O error during a read attempt.
 *                   This error is not recoverable.
 * The first function to detect the error will report it, while attempting to
 * process the item it applies to.  In other words, if greadgksm encounters a
 * file error while trying to go on to the next item after successfully reading
 * the current item, the error will not be reported until the next get/read
 * call.  After a fatal error has been reported (via GKS error 163, item is
 * invalid), subsequent get/read attempts will return error 162, no items left
 * in MI, since the error is unrecoverable and no more reading is allowed.
 */
    int
greadgksm(Gint ws_id, Gint length, char *record)
                          	/* workstation identifier */
        	           	/* maximum item data record length */
                           	/* input metafile item data-record */
{
    int             istat;
    WS_STATE_PTR   ws;

    GKSERROR((xgks_state.gks_state != GWSOP && xgks_state.gks_state !=
		GWSAC && xgks_state.gks_state != GSGOP),
	     7,
	     errgreadgksm);

    /* check for invalid workstation id */
    GKSERROR((!VALID_WSID(ws_id)), 20, errgreadgksm);

    GKSERROR(((ws = OPEN_WSID(ws_id)) == NULL), 25, errgreadgksm);

    GKSERROR((WS_CAT(ws) != GMI), 34, errgreadgksm);

    if (ws->mf.any->CurItem.type == 0)
	ws->mf.any->GksmEmpty = TRUE;

    GKSERROR((ws->mf.any->GksmEmpty == TRUE), 162, errgreadgksm);

    if (ws->mf.any->filestat == MF_FILE_ERR) {
	ws->mf.any->GksmEmpty = TRUE;
	(void) gerrorhand(162, errgreadgksm, xgks_state.gks_err_file);
	return 162;
    }
    if (XgksValidGksMItem(ws->mf.any->CurItem.type) == INVALID)
	ws->mf.any->filestat = MF_ITEM_ERR;

    GKSERROR(((ws->mf.any->filestat == MF_ITEM_ERR) && (length != 0)),
	     MF_ITEM_ERR, errgreadgksm);

    GKSERROR(((ws->mf.any->filestat == MF_DATA_ERR) && (length != 0)),
	     MF_DATA_ERR, errgreadgksm);

    GKSERROR((length < 0), 166, errgreadgksm);

    if ((istat = MI_READ_ITEM(&ws->mf, record)) != 0)
	return istat;

    ws->mf.any->filestat	= MI_GET_NEXT_ITEM(&ws->mf);

    return OK;
}


/*
 * Return the minimum possible size for the data associated with a GKSM 
 * item-type.
 *
 * Note, since this routine is used by ginterpret(), we return the size of the 
 * decoded, binary structures -- not the file-encoded size.
 */
    static int
recSize(Gint type)
{
    switch (type) {

    case 0:
    case 2:
    case 82:
	return 0;

    case 1:
    case 3:
    case 21:
    case 22:
    case 24:
    case 25:
    case 26:
    case 28:
    case 29:
    case 33:
    case 35:
    case 37:
    case 38:
    case 39:
    case 40:
    case 44:
    case 81:
    case 84:
	return sizeof(XGKSMONE);

    case 4:
    case 30:
    case 36:
    case 83:
    case 92:
    case 93:
    case 95:
	return sizeof(XGKSMTWO);

    case 5:
	return sizeof(XGKSMONE);

    case 11:
    case 12:
    case 14:
	return sizeof(XGKSMONE);

    case 13:
	return sizeof(XGKSMTEXT);

    case 15:
	return sizeof(XGKSMCELLARRAY);

    case 23:
    case 27:
    case 31:
    case 32:
	return sizeof(XGKSMSIZE);

    case 34:
	return sizeof(XGKSMCHARVEC);

    case 43:
	return sizeof(XGKSMASF);

    case 41:
	return sizeof(XGKSMPATSIZ);

    case 42:
	return sizeof(XGKSMPATREF);

    case 51:
    case 52:
	return sizeof(XGKSMLMREP);

    case 53:
	return sizeof(XGKSMTEXTREP);

    case 54:
	return sizeof(XGKSMFILLREP);

    case 55:
	return sizeof(XGKSMPATREP);

    case 56:
	return sizeof(XGKSMCOLOURREP);

    case 61:
    case 71:
    case 72:
	return sizeof(XGKSMLIMIT);

    case 91:
	return sizeof(XGKSMSEGTRAN);

    case 94:
	return sizeof(XGKSMSEGPRI);

    default:
	return INVALID;
    }
}


/*
 * INTERPRET GKSM ITEM
 */
    int
ginterpret(Ggksmit *recInfo, char *data)
                            	/* item type and length */
                         	/* item data-record */
{
    GKSERROR((xgks_state.gks_state != GWSOP && xgks_state.gks_state !=
		GWSAC && xgks_state.gks_state != GSGOP),
	     7,
	     errginterpret);

    GKSERROR((recInfo == NULL), 163, errginterpret);

    GKSERROR(((recInfo->length > 0) && (data == NULL)), 165, errginterpret);

    /*
     * We no longer check for invalid size (error 161) because the GKSM
     * backend returns the length of the formatted data record rather than the
     * size of the internally-used binary structures.
     */
#if 0
    GKSERROR((recInfo->length < recSize(recInfo->type)),
	     161, errginterpret);
#endif

    GKSERROR((XgksValidGksMItem(recInfo->type) == INVALID), 164,
	     errginterpret);

    /*
     * Can't check for 165 in ginterpret due to file format.
     * Can't really check for 163, either.
     */

    GKSERROR((recInfo->type > 100), 167, errginterpret);

    GKSERROR((NOTSUPPORTED(recInfo->type)), 168, errginterpret);

    GKSERROR((XgksExecData(recInfo->type, data) != 0), 164, errginterpret);

    return OK;
}


/*
 * Open an input Metafile: scan header and get first item.
 */
    int
XgksMiOpenWs(ws)
    WS_STATE_PTR    ws;
{
    int		status	= 26;		/* return status = workstation cannot
					 * be opened */

    ws->wscolour	= MAX_META_WSCOLOURS;
    ws->set_colour_rep	= NULL;

    if ((strstr(ws->conn, ".cgm") == NULL
	    ? GMmiOpen(&ws->mf, ws->conn)
	    : CGMmiOpen(&ws->mf, ws->conn))
	== OK) {

	ws->mf.any->filestat		= MI_GET_NEXT_ITEM(&ws->mf);

	if (ws->mf.any->filestat == METAFILE_OK) {
	    status	= OK;
	} else {
	    MI_CLOSE(&ws->mf);
	}
    }

    return status;
}


/*
 * Open an output Metafile.
 */
    int
XgksMoOpenWs(ws)
    WS_STATE_PTR    ws;
{
  Gint status;
    ws->wscolour	= MAX_META_WSCOLOURS;
    ws->set_colour_rep	= NULL;
    
    /* Make sure window boundaries are set to zero to avoid */
    /* floating point exceptions on DEC OSF *js* 9.97 */
    ws->wbound.x = 0;
    ws->wbound.y = 0;
      
    if (strstr(ws->conn, ".cgm") != NULL || strstr(ws->conn, ".CGM") != NULL) {
      status = CGMmoOpen(ws);
    } else if (strstr(ws->conn, ".ps") != NULL ||
	       strstr(ws->conn, ".PS") != NULL){
      status = PSmoOpen(ws);
    } else if (strstr(ws->conn, ".gif") != NULL ||
	       strstr(ws->conn, ".GIF") != NULL){
      status = GIFmoOpen(ws);
    } else {
      status = GMmoOpen(ws);
    }
    return status == OK ? OK :26;
}

/*
 * Close an input Metafile.
 */
    int
XgksMiCloseWs(WS_STATE_PTR ws)
{
    (void) fclose(ws->mf.any->fp);

    return OK;
}


/*
 * Close an output Metafile.
 */
    int
XgksMoCloseWs(WS_STATE_PTR ws)
{
  return MO_CLOSE(&ws->mf);
}


/*
 * Set the clear flag in an output Metafile.
 */
    int
XgksMoClearWs(WS_STATE_PTR ws, Gclrflag flag)
{
    Metafile       *mf	= &ws->mf;

    return MO_CLEAR(mf, 1, flag);
}


/*
 * Redraw all segments in an output Metafile.
 */
    int
XgksMoReDrawAllSeg(WS_STATE_PTR ws)
{
    Metafile       *mf	= &ws->mf;

    return MO_REDRAW_ALL_SEG(mf, 1);
}


/*
 * Set the update flag in an output Metafile.
 */
    int
XgksMoUpdateWs(WS_STATE_PTR ws, Gregen regenflag)
{
    Metafile       *mf	= &ws->mf;

    return MO_UPDATE(mf, 1, regenflag);
}


/*
 * Set the deferal state in an output Metafile.
 */
    int
XgksMoDeferWs(WS_STATE_PTR ws, Gdefmode defer_mode, Girgmode regen_mode)
{
    Metafile       *mf	= &ws->mf;

    return MO_DEFER(mf, 1, defer_mode, regen_mode);
}


/*
 * Write a message to an output Metafile.
 */
    int
XgksMoMessage(WS_STATE_PTR ws, Gchar *string)
{
    Metafile       *mf	= &ws->mf;

    return MO_MESSAGE(mf, 1, string);
}


/*
 * Write a graphic to an output Metafile.
 */
    int
XgksMoGraphicOutputToWs(WS_STATE_PTR ws, Gint code, Gint num_pt, Gpoint *pos)
{
    Metafile       *mf	= &ws->mf;

    return MO_GRAPHIC(mf, 1, code, num_pt, pos);
}


/*
 * Write a graphic to all, appropriate, output Metafiles.
 */
    int
XgksMoGraphicOutput(Gint code, Gint num_pt, Gpoint *pos)
{
    if (num_gksmo > 0)
	GMoutputGraphic(active_gksmo, num_gksmo, code, num_pt, pos);
    if (num_cgmo > 0)
	CGMoutputGraphic(active_cgmo, num_cgmo, code, num_pt, pos);
    if (num_psmo > 0)
	PSoutputGraphic(active_psmo, num_psmo, code, num_pt, pos);
    if (num_gifmo > 0)
	GIFoutputGraphic(active_gifmo, num_gifmo, code, num_pt, pos);

    return OK;
}


/*
 * Write text to an output Metafile.
 */
    int
XgksMoTextToWs(WS_STATE_PTR ws, Gpoint *at, Gchar *string)
{
    Metafile       *mf	= &ws->mf;

    return MO_TEXT(mf, 1, at, string);
}


/*
 * Write text to all, appropriate, output Metafiles.
 */
    int
XgksMoText(Gpoint *at, Gchar *string)
{
    if (num_gksmo > 0)
	GMtext(active_gksmo, num_gksmo, at, string);
    if (num_cgmo > 0)
	CGMtext(active_cgmo, num_cgmo, at, string);
    if (num_psmo > 0)
	PStext(active_psmo, num_psmo, at, string);
    if (num_gifmo > 0)
	GIFtext(active_gifmo, num_gifmo, at, string);

    return OK;
}


/*
 * Write a cell array to an output Metafile.
 */
    int
XgksMoCellArrayToWs(WS_STATE_PTR ws, Gpoint *ll, Gpoint *ur, Gpoint *lr, Gint row, Gint *colour, Gipoint *dim)
{
    Metafile       *mf	= &ws->mf;

    return MO_CELL_ARRAY(mf, 1, ll, ur, lr, row, colour, dim);
}


/*
 * Write a cell array to all, appropriate, output Metafiles.
 */
    int
XgksMoCellArray(Gpoint *ll, Gpoint *ur, Gpoint *lr, Gint row, Gint *colour, Gipoint *dim)
{
    if (num_gksmo > 0)
	GMcellArray(active_gksmo, num_gksmo, ll, ur, lr, row, 
				 colour, dim);
    if (num_cgmo > 0)
	CGMcellArray(active_cgmo, num_cgmo, ll, ur, lr, row, 
				 colour, dim);
    if (num_psmo > 0)
	PScellArray(active_psmo, num_psmo, ll, ur, lr, row, 
				 colour, dim);
    if (num_gifmo > 0)
	GIFcellArray(active_gifmo, num_gifmo, ll, ur, lr, row, 
				 colour, dim);

    return OK;
}


/*
 * Set the size of graphics in an output Metafile.
 */
    int
XgksMoSetGraphicSizeOnWs(WS_STATE_PTR ws, Gint code, double size)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_GRAPH_SIZE(mf, 1, code, size);
}


/*
 * Set the size of graphics in all, appropriate, output Metafiles.
 */
    int
XgksMoSetGraphicSize(Gint code, double size)
{

    if (num_gksmo > 0)
	GMsetGraphSize(active_gksmo, num_gksmo, code, size);
    if (num_cgmo > 0)
	CGMsetGraphSize(active_cgmo, num_cgmo, code, size);
    if (num_psmo > 0)
	PSsetGraphSize(active_psmo, num_psmo, code, size);
    if (num_gifmo > 0)
	GIFsetGraphSize(active_gifmo, num_gifmo, code, size);

    return OK;
}


/*
 * Close the open segment in an output Metafile.
 */
    int
XgksMoCloseSegOnWs(WS_STATE_PTR ws)
{
    Metafile       *mf	= &ws->mf;

    return MO_CLOSE_SEG(mf, 1);
}


/*
 * Close the open segment in all, appropriate, output Metafiles.
 */
    int
XgksMoCloseSeg(void)
{
    if (num_gksmo > 0)
	GMcloseSeg(active_gksmo, num_gksmo);
    if (num_cgmo > 0)
	CGMcloseSeg(active_cgmo, num_cgmo);
    if (num_psmo > 0)
	PScloseSeg(active_psmo, num_psmo);
    if (num_gifmo > 0)
	GIFcloseSeg(active_gifmo, num_gifmo);

    return OK;
}


/*
 * Set the graphic attributes in an output Metafile.
 */
    int
XgksMoSetGraphicAttrOnWs(WS_STATE_PTR ws, Gint code, Gint attr)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_GRAPH_ATTR(mf, 1, code, attr);
}


/*
 * Set the graphic attributes in all, appropriate, output Metafiles.
 */
    int
XgksMoSetGraphicAttr(Gint code, Gint attr)
{
    if (num_gksmo > 0)
	GMsetGraphAttr(active_gksmo, num_gksmo, code, attr);
    if (num_cgmo > 0)
	CGMsetGraphAttr(active_cgmo, num_cgmo, code, attr);
    if (num_psmo > 0)
	PSsetGraphAttr(active_psmo, num_psmo, code, attr);
    if (num_gifmo > 0)
	GIFsetGraphAttr(active_gifmo, num_gifmo, code, attr);

    return OK;
}


/*
 * Set the font precision in an output Metafile.
 */
    int
XgksMoSetTextFPOnWs(WS_STATE_PTR ws, Gtxfp *txfp)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_TEXT_FP(mf, 1, txfp);
}


/*
 * Set the font precision in all, appropriate, output Metafiles.
 */
    int
XgksMoSetTextFP(Gtxfp *txfp)
{
    if (num_gksmo > 0)
	GMsetTextFP(active_gksmo, num_gksmo, txfp);
    if (num_cgmo > 0)
	CGMsetTextFP(active_cgmo, num_cgmo, txfp);
    if (num_psmo > 0)
	PSsetTextFP(active_psmo, num_psmo, txfp);
    if (num_gifmo > 0)
	GIFsetTextFP(active_gifmo, num_gifmo, txfp);

    return OK;
}


/*
 * Set the character up-vector in an output Metafile.
 */
    int
XgksMoSetCharUpOnWs(WS_STATE_PTR ws, Gpoint *up, Gpoint *base)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_CHAR_UP(mf, 1, up, base);
}


/*
 * Set the character up-vector in all, appropriate, output Metafiles.
 */
    int
XgksMoSetCharUp(void)
{
    if (num_gksmo > 0)
	GMsetCharUp(active_gksmo, num_gksmo, (Gpoint*)NULL,
				      (Gpoint*)NULL);
    if (num_cgmo > 0)
	CGMsetCharUp(active_cgmo, num_cgmo, (Gpoint*)NULL,
				      (Gpoint*)NULL);
    if (num_psmo > 0)
	PSsetCharUp(active_psmo, num_psmo, &xgks_state.gks_chattr.up,
				      &xgks_state.gks_chattr.base);
    if (num_gifmo > 0)
	GIFsetCharUp(active_gifmo, num_gifmo, &xgks_state.gks_chattr.up,
				      &xgks_state.gks_chattr.base);

    return OK;
}


/*
 * Set the text-path in an output Metafile.
 */
    int
XgksMoSetTextPathOnWs(WS_STATE_PTR ws, Gtxpath path)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_TEXT_PATH(mf, 1, path);
}


/*
 * Set the text-path in all, appropriate, output Metafiles.
 */
    int
XgksMoSetTextPath(Gtxpath path)
{
    if (num_gksmo > 0)
	GMsetTextPath(active_gksmo, num_gksmo, path);
    if (num_cgmo > 0)
	CGMsetTextPath(active_cgmo, num_cgmo, path);
    if (num_psmo > 0)
	PSsetTextPath(active_psmo, num_psmo, path);
    if (num_gifmo > 0)
	GIFsetTextPath(active_gifmo, num_gifmo, path);

    return OK;
}


/*
 * Set the text-alignment in an output Metafile.
 */
    int
XgksMoSetTextAlignOnWs(WS_STATE_PTR ws, Gtxalign *align)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_TEXT_ALIGN(mf, 1, align);
}


/*
 * Set the text-alignment in all, appropriate, output Metafiles.
 */
    int
XgksMoSetTextAlign(Gtxalign *align)
{
    if (num_gksmo > 0)
	GMsetTextAlign(active_gksmo, num_gksmo, align);
    if (num_cgmo > 0)
	CGMsetTextAlign(active_cgmo, num_cgmo, align);
    if (num_psmo > 0)
	PSsetTextAlign(active_psmo, num_psmo, align);
    if (num_gifmo > 0)
	GIFsetTextAlign(active_gifmo, num_gifmo, align);

    return OK;
}


/*
 * Set the interior fill-style in an output Metafile.
 */
    int
XgksMoSetFillIntStyleOnWs(WS_STATE_PTR ws, Gflinter style)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_FILL_STYLE(mf, 1, style);
}


/*
 * Set the interior fill-style in all, appropriate, output Metafiles.
 */
    int
XgksMoSetFillIntStyle(Gflinter style)
{
    if (num_gksmo > 0)
	GMsetFillStyle(active_gksmo, num_gksmo, style);
    if (num_cgmo > 0)
	CGMsetFillStyle(active_cgmo, num_cgmo, style);
    if (num_psmo > 0)
	PSsetFillStyle(active_psmo, num_psmo, style);
    if (num_gifmo > 0)
	GIFsetFillStyle(active_gifmo, num_gifmo, style);

    return OK;
}


/*
 * Set the pattern size in an output Metafile.
 */
    int
XgksMoSetPatSizeOnWs(WS_STATE_PTR ws)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_PATTERN_SIZE(mf, 1);
}


/*
 * Set the pattern size in all, appropriate, output Metafiles.
 */
    int
XgksMoSetPatSize(void)
{
    if (num_gksmo > 0)
	GMsetPatSize(active_gksmo, num_gksmo);
    if (num_cgmo > 0)
	CGMsetPatSize(active_cgmo, num_cgmo);
    if (num_psmo > 0)
	PSsetPatSize(active_psmo, num_psmo);
    if (num_gifmo > 0)
	GIFsetPatSize(active_gifmo, num_gifmo);

    return OK;
}


/*
 * Set the pattern reference-point in an output Metafile.
 */
    int
XgksMoSetPatRefOnWs(WS_STATE_PTR ws)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_PATTERN_REFPT(mf, 1);
}


/*
 * Set the pattern reference-point in all, appropriate, output Metafiles.
 */
    int
XgksMoSetPatRef(void)
{
    if (num_gksmo > 0)
	GMsetPatRefpt(active_gksmo, num_gksmo);
    if (num_cgmo > 0)
	CGMsetPatRefpt(active_cgmo, num_cgmo);
    if (num_psmo > 0)
	PSsetPatRefpt(active_psmo, num_psmo);
    if (num_gifmo > 0)
	GIFsetPatRefpt(active_gifmo, num_gifmo);

    return OK;
}


/*
 * Set the ASF in an output Metafile.
 */
    int
XgksMoSetAsfOnWs(WS_STATE_PTR ws)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_ASF(mf, 1);
}


/*
 * Set the ASF in all, appropriate, output Metafiles.
 */
    int
XgksMoSetAsf(void)
{
    if (num_gksmo > 0)
	GMsetAsf(active_gksmo, num_gksmo);
    if (num_cgmo > 0)
	CGMsetAsf(active_cgmo, num_cgmo);
    if (num_psmo > 0)
	PSsetAsf(active_psmo, num_psmo);
    if (num_gifmo > 0)
	GIFsetAsf(active_gifmo, num_gifmo);

    return OK;
}


/*
 * Set the line and marker representation in an output Metafile.
 */
    int
XgksMoSetLineMarkRep(WS_STATE_PTR ws, Gint code, Gint idx, Gint type, double size, Gint colour)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_LINE_MARKER_REP(mf, 1, code, idx, type, size, colour);
}


/*
 * Set the text representation in an output Metafile.
 */
    int
XgksMoSetTextRep(WS_STATE_PTR ws, Gint idx, Gtxbundl *rep)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_TEXT_REP(mf, 1, idx, rep);
}


/*
 * Set the fill representation in an output Metafile.
 */
    int
XgksMoSetFillRep(WS_STATE_PTR ws, Gint idx, Gflbundl *rep)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_FILL_REP(mf, 1, idx, rep);
}


/*
 * Set the pattern representation in an output Metafile.
 */
    int
XgksMoSetPatRep(WS_STATE_PTR ws, Gint idx, Gptbundl *rep)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_PATTERN_REP(mf, 1, idx, rep);
}


/*
 * Set the colour representation in an output Metafile.
 */
    int
XgksMoSetColourRep(WS_STATE_PTR ws, Gint idx, Gcobundl *rep)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_COLOUR_REP(mf, 1, idx, rep);
}


/*
 * Set the clipping rectangle in an output Metafile.
 */
    int
XgksMoSetClipOnWs(WS_STATE_PTR ws, Glimit *rect)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_CLIPPING(mf, 1, rect);
}


/*
 * Set the clipping rectangle in all, appropriate, output Metafiles.
 */
    int
XgksMoSetClip(Glimit *rect)
{
    if (num_gksmo > 0)
	GMsetClip(active_gksmo, num_gksmo, rect);
    if (num_cgmo > 0)
	CGMsetClip(active_cgmo, num_cgmo, rect);
    if (num_psmo > 0)
	PSsetClip(active_psmo, num_psmo, rect);
    if (num_gifmo > 0)
	GIFsetClip(active_gifmo, num_gifmo, rect);

    return OK;
}


/*
 * Set the viewport limits in an output Metafile.
 */
    int
XgksMoSetLimit(WS_STATE_PTR ws, Gint code, Glimit *rect)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_LIMIT(mf, 1, code, rect);
}


/*
 * Rename a segment in all, appropriate, output Metafiles.
 */
    int
XgksMoRenameSeg(Gint old, Gint new)
{
    if (num_gksmo > 0)
	GMrenameSeg(active_gksmo, num_gksmo, old, new);
    if (num_cgmo > 0)
	CGMrenameSeg(active_cgmo, num_cgmo, old, new);
    if (num_psmo > 0)
	PSrenameSeg(active_psmo, num_psmo, old, new);
    if (num_gifmo > 0)
	GIFrenameSeg(active_gifmo, num_gifmo, old, new);

    return OK;
}


/*
 * Set the segment transformation in an output Metafile.
 */
    int
XgksMoSetSegTransOnWs(WS_STATE_PTR ws, Gint name, Gfloat (*matrix)[])
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_SEG_TRANS(mf, 1, name, matrix);
}


/*
 * Set the segment transformation in all, appropriate, output Metafiles.
 */
    int
XgksMoSetSegTrans(Gint name, Gfloat (*matrix)[])
{
    if (num_gksmo > 0)
	GMsetSegTran(active_gksmo, num_gksmo, name, matrix);
    if (num_cgmo > 0)
	CGMsetSegTran(active_cgmo, num_cgmo, name, matrix);
    if (num_psmo > 0)
	PSsetSegTran(active_psmo, num_psmo, name, matrix);
    if (num_gifmo > 0)
	GIFsetSegTran(active_gifmo, num_gifmo, name, matrix);

    return OK;
}


/*
 * Set the segment attributes in an output Metafile.
 */
    int
XgksMoSetSegAttrOnWs(WS_STATE_PTR ws, Gint name, Gint code, Gint attr)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_SEG_ATTR(mf, 1, name, code, attr);
}


/*
 * Set the segment visibility in all, appropriate, output Metafiles.
 */
    int
XgksMoSetSegVis(Gint name, Gsegvis vis)
{
    if (num_gksmo > 0)
	GMsetSegVis(active_gksmo, num_gksmo, name, vis);
    if (num_cgmo > 0)
	CGMsetSegVis(active_cgmo, num_cgmo, name, vis);
    if (num_psmo > 0)
	PSsetSegVis(active_psmo, num_psmo, name, vis);
    if (num_gifmo > 0)
	GIFsetSegVis(active_gifmo, num_gifmo, name, vis);

    return OK;
}


/*
 * Set the segment highlighting in all, appropriate, output Metafiles.
 */
    int
XgksMoSetSegHiLight(Gint name, Gseghi hilight)
{
    if (num_gksmo > 0)
	GMsetSegHilight(active_gksmo, num_gksmo, name, hilight);
    if (num_cgmo > 0)
	CGMsetSegHilight(active_cgmo, num_cgmo, name, hilight);
    if (num_psmo > 0)
	PSsetSegHilight(active_psmo, num_psmo, name, hilight);
    if (num_gifmo > 0)
	GIFsetSegHilight(active_gifmo, num_gifmo, name, hilight);

    return OK;
}


/*
 * Set the segment priority in an output Metafile.
 */
    int
XgksMoSetSegPriOnWs(WS_STATE_PTR ws, Gint name, double pri)
{
    Metafile       *mf	= &ws->mf;

    return MO_SET_SEG_PRI(mf, 1, name, pri);
}


/*
 * Set the segment priority in all, appropriate output Metafiles.
 */
    int
XgksMoSetSegPri(Gint name, double pri)
{
    if (num_gksmo > 0)
	GMsetSegPri(active_gksmo, num_gksmo, name, pri);
    if (num_cgmo > 0)
	CGMsetSegPri(active_cgmo, num_cgmo, name, pri);
    if (num_psmo > 0)
	PSsetSegPri(active_psmo, num_psmo, name, pri);
    if (num_gifmo > 0)
	GIFsetSegPri(active_gifmo, num_gifmo, name, pri);

    return OK;
}


/*
 * Set segment detectability in all, appropriate output Metafiles.
 */
    int
XgksMoSetSegDet(Gint name, Gsegdet det)
{
    if (num_gksmo > 0)
	GMsetSegDetect(active_gksmo, num_gksmo, name, det);
    if (num_cgmo > 0)
	CGMsetSegDetect(active_cgmo, num_cgmo, name, det);
    if (num_psmo > 0)
	PSsetSegDetect(active_psmo, num_psmo, name, det);
    if (num_gifmo > 0)
	GIFsetSegDetect(active_gifmo, num_gifmo, name, det);

    return OK;
}


/*
 * Add an output Metafile to a list of active, output Metafiles.
 */
    static void
add_mo(Metafile *mo, Metafile *list, int *num)
{
    assert(*num >= 0);
    assert(*num < MAX_ACTIVE_WS);

    list[(*num)++]	= *mo;
}


/*
 * Remove an output Metafile from a list of active, output Metafiles.
 */
    static void
remove_mo(mo, list, num)
    Metafile	*mo;
    Metafile	*list;
    int		*num;
{
    Metafile	*outp;

    assert(*num > 0);
    assert(*num <= MAX_ACTIVE_WS);

    /* Find the Metafile to be removed. */
    for (outp = list + *num; list < outp; ++list)
	if (list->any == mo->any)
	    break;

    assert(list < outp);

    if (list < outp) {

	/* Shift the list down over the found Metafile. */
	for (--outp; list < outp; ++list)
	    list[0].any	= list[1].any;

	--*num;
    }
}


/*
 * Activate an output Metafile: add it to the list of active, output
 * Metafiles and write initial output Metafile attributes.
 */
    int
XgksMoActivateWs(WS_STATE_PTR ws)
{
    switch(ws->mf.any->type){
    case MF_GKSM:
      add_mo(&ws->mf, active_gksmo, &num_gksmo);
      break;
    case MF_PS:
      add_mo(&ws->mf, active_psmo, &num_psmo);
      break;
    case MF_GIF:
      add_mo(&ws->mf, active_gifmo, &num_gifmo);
      break;
    case MF_CGM:
    default:
      add_mo(&ws->mf, active_cgmo, &num_cgmo);
      break;
    }
    
    XgksMoSetClipOnWs(ws, &xgks_state.cliprec.rec);
    
    XgksSetLineAttrMo(ws, &xgks_state.gks_lnattr);
    XgksSetMarkAttrMo(ws, &xgks_state.gks_mkattr);
    XgksSetTextAttrMo(ws, &xgks_state.gks_txattr, &xgks_state.gks_chattr);
    XgksMoSetCharUpOnWs(ws, (Gpoint *) NULL, (Gpoint *) NULL);
    XgksSetFillPatAttrMo(ws, &xgks_state.gks_flattr, &xgks_state.gks_ptattr);
    
    XgksMoSetPatSizeOnWs(ws);
    XgksMoSetPatRefOnWs(ws);
    XgksMoSetAsfOnWs(ws);
    XgksMoSetGraphicAttrOnWs(ws, 44, xgks_state.gks_pick_id);
    
    return OK;
}


/*
 * Deactivate an output Metafile: remove it from the list of active, output
 * Metafiles.
 */
    int
XgksMoDeactivateWs(WS_STATE_PTR ws)
{
    switch(ws->mf.any->type){
      case MF_GKSM:
	remove_mo(&ws->mf, active_gksmo, &num_gksmo);
	break;
      case MF_PS:
	remove_mo(&ws->mf, active_psmo, &num_psmo);
	break;
      case MF_GIF:
	remove_mo(&ws->mf, active_gifmo, &num_gifmo);
	break;
      case MF_CGM:
      default:
	remove_mo(&ws->mf, active_cgmo, &num_cgmo);
	break;
    }
    return OK;
}


/*
 * Initialize the Metafile system.
 */
    int
XgksInitGksM(void)
{
    return OK;
}
