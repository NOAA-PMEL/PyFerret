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
 * FORTRAN to C binding for XGKS
 *
 * David Berkowitz
 * TCS Development
 * Cambridge MA
 *
 * May 30 1988
 *
 * $Id$
 * $__Header$
 */

#ifndef FORTXGKS_H_INCLUDED
#define FORTXGKS_H_INCLUDED

#include <stdio.h>		/* for FILE */
#include "fortmac.h"

#ifdef DEBUG
#   define debug(f) (fileno(stdout)>0 ? (printf f,fflush(stdout)) : 0)
#else
#   define debug(f)
#endif

#ifndef	NULL
#   define NULL 0
#endif

#define MAGICNUMBER 7
#define FORTSTDIN 5
#define FORTSTDOUT 6

extern int      retval;			/* return value from C calls */
extern int      global_errfil;		/* global error file unit number */
extern FILE    *errfp;			/* error file */

typedef enum {
    errgurec,
    errgprec
}               Fort_errmap;

/* Fort_Gredraw - workstation redraw state */
typedef enum {
    FORT_GRD_GKS,			/* redraw due to GKS */
    FORT_GRD_X				/* redraw due to X */
}               Fort_Gredraw;

/* Fort_Gacf - attribute control flag */
typedef enum {
    FORT_GCURNT,
    FORT_GSPEC
}               Fort_Gacf;

/* Fort_Gasf - aspect source flag */
typedef enum {
    FORT_GBUNDL,
    FORT_GINDIV
}               Fort_Gasf;

/* Fort_Gattrs - attributes used */
typedef enum {
    FORT_GPLATT,
    FORT_GPMATT,
    FORT_GTXATT,
    FORT_GFAATT
}               Fort_Gattrs;

/* Fort_Gprflag - prompt flag */
typedef enum {
    FORT_GPROFF,
    FORT_GPRON
}               Fort_Gprflag;

/* Fort_Gimode - input mode */
typedef enum {
    FORT_GREQU,
    FORT_GSAMPL,
    FORT_GEVENT
}               Fort_Gimode;

/* Fort_Gesw - echo switch */
typedef enum {
    FORT_GNECHO,
    FORT_GECHO
}               Fort_Gesw;

/* Fort_Gclip - clipping indicator */
typedef enum {
    FORT_GNCLIP,
    FORT_GCLIP
}               Fort_Gclip;

/* Fort_Gclrflag - clear control flag */
typedef enum {
    FORT_GCONDI,
    FORT_GALWAY
}               Fort_Gclrflag;

/* Fort_Gcoavail - color availablity */
typedef enum {
    FORT_GMONOC,
    FORT_GCOLOR
}               Fort_Gcoavail;

/* Fort_Gcovaild - color values valid */
typedef enum {
    FORT_GABSNT,
    FORT_GPRSNT
}               Fort_Gcovalid;

/* Fort_Gcsw - coordinate switch */
typedef enum {
    FORT_GWC,				/* world coordinates */
    FORT_GNDC				/* normalized device coordinates */
}               Fort_Gcsw;

/* Fort_Gdefmode - deferral mode */
typedef enum {
    FORT_GASAP,				/* as soon as possible */
    FORT_GBNIG,				/* before next interaction globally */
    FORT_GBNIL,				/* before next interaction locally */
    FORT_GASTI				/* at some time in */
}               Fort_Gdefmode;

/* Fort_Girgmode - implicit regeneration mode */
typedef enum {
    FORT_GSUPPD,
    FORT_GALLOW
}               Fort_Girgmode;

/* Fort_Gpfcf - polyline/fill area flag */
typedef enum {
    FORT_GPLINE,
    FORT_GFILLA
}               Fort_Gpfcf;

/* Fort_Gflinter - fill area interior style */
typedef enum {
    FORT_GHOLLO,
    FORT_GSOLID,
    FORT_GPATTR,
    FORT_GHATCH
}               Fort_Gflinter;

/* Fort_Gdevunits - device coordinate units */
typedef enum {
    FORT_GMETRE,
    FORT_GOTHU
}               Fort_Gdevunits;

/* Fort_Gdspsurf - display surface */
typedef enum {
    FORT_GNEMPT,
    FORT_GEMPTY
}               Fort_Gdspsurf;

/* Fort_Giclass - input class */
typedef enum {
    FORT_GNCLAS,
    FORT_GLOCAT,
    FORT_GSTROK,
    FORT_GVALUA,
    FORT_GCHOIC,
    FORT_GPICK,
    FORT_GSTRIN
}               Fort_Giclass;

/* Fort_Gtxprec - text precision */
typedef enum {
    FORT_GSTRP,
    FORT_GCHARP,
    FORT_GSTRKP
}               Fort_Gtxprec;

/* Fort_Ginqtype - inquiry type */
typedef enum {
    FORT_GSET,
    FORT_GREALI
}               Fort_Ginqtype, Fort_Gqtype;	/* two names due to an error

						 * in the C Binding */

/* Fort_Gistat - input status */
typedef enum {
    FORT_GNONE,
    FORT_GOK,
    FORT_GNPICK,
    FORT_GNCHOI = 2
} Fort_Gistat;

/* Fort_Glevel - level of GKS */
typedef enum {
    FORT_GLMA = -3,
    FORT_GLMB,
    FORT_GLMC,
    FORT_GL0A,
    FORT_GL0B,
    FORT_GL0C,
    FORT_GL1A,
    FORT_GL1B,
    FORT_GL1C,
    FORT_GL2A,
    FORT_GL2B,
    FORT_GL2C
}               Fort_Glevel;

/* Fort_Gmodtype - dynamic modification type */
typedef enum {
    FORT_GIRG,
    FORT_GIMM
}               Fort_Gmodtype;

/* Fort_Gnframe - new frame action at update */
/* Fort_Gstore - workstation storage of non-segment primitives state */
typedef enum {
    FORT_GNO,
    FORT_GYES
}               Fort_Gnframe, Fort_Gstore;

/* Fort_Gos - GKS operating state */
typedef enum {
    FORT_GGKCL,				/* GKS closed */
    FORT_GGKOP,				/* GKS open */
    FORT_GWSOP,				/* workstation open */
    FORT_GWSAC,				/* workstation active */
    FORT_GSGOP				/* segment open */
}               Fort_Gos;

/* Fort_Gtxpath - text path */
typedef enum {
    FORT_GRIGHT,
    FORT_GLEFT,
    FORT_GUP,
    FORT_GDOWN
}               Fort_Gtxpath;

/* Fort_Gtxhor - text alignment horizontal component */
typedef enum {
    FORT_GAHNOR,
    FORT_GALEFT,
    FORT_GACENT,
    FORT_GARITE
}               Fort_Gtxhor;

/* Fort_Gtxver - text alignment vertical component */
typedef enum {
    FORT_GAVNOR,
    FORT_GATOP,
    FORT_GACAP,
    FORT_GAHALF,
    FORT_GABASE,
    FORT_GABOTT
}               Fort_Gtxver;

/* Fort_Gregen - regeneration flag */
typedef enum {
    FORT_GPOSTP,
    FORT_GPERFO
}               Fort_Gregen;

/* Fort_Gsegdet - segment detectability */
typedef enum {
    FORT_GUNDET,
    FORT_GDETEC
}               Fort_Gsegdet;

/* Fort_Gseghi - segment highlighting */
typedef enum {
    FORT_GNORML,
    FORT_GHILIT
}               Fort_Gseghi;

/* Fort_Gsegvis - segment visibility */
typedef enum {
    FORT_GINVIS,
    FORT_GVISI
}               Fort_Gsegvis;

/* Fort_Gsimultev - simultaneous events */
typedef enum {
    FORT_GNMORE,
    FORT_GMORE
}               Fort_Gsimultev;

/* Fort_Gvpri - viewport input priority */
typedef enum {
    FORT_GHIGHR,
    FORT_GLOWER
}               Fort_Gvpri;

/* Fort_Gwscat - workstation category */
typedef enum {
    FORT_GOUTPT,
    FORT_GINPUT,
    FORT_GOUTIN,
    FORT_GWISS,
    FORT_GMO,
    FORT_GMI
}               Fort_Gwscat;

/* Fort_Gwsclass - workstation classification */
typedef enum {
    FORT_GVECTR,
    FORT_GRASTR,
    FORT_GOTHWK
}               Fort_Gwsclass;

/* Fort_Gwsstate - workstation state */
typedef enum {
    FORT_GINACT,
    FORT_GACTIV
}               Fort_Gwsstate;

/* Fort_Gwstus - workstation transformation update state */
typedef enum {
    FORT_GNPEND,
    FORT_GPEND
}               Fort_Gwstus;

#endif	/* FORTXGKS_H_INCLUDED */
