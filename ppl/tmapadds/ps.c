/*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*
*/



/*
 * PostScript driver for XGKS metafiles
 * Created by Joe Sirott, Pacific Marine Environmental Lab
 *
 */

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
 */

/*LINTLIBRARY*/

#include <math.h>
#include <stdlib.h>
#include <time.h>		/* for time(), localtime(), and strftime() */
#include <sys/types.h>		/* for uid_t */
#include <sys/utsname.h>	/* for uname() */
#include <unistd.h>		/* for getlogin() */
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <stdarg.h>
#include "udposix.h"
#include "gks_implem.h"
#include "cgm/cgm.h"		/* for public, API details */
#include "cgm/cgm_implem.h"		/* for implementation details */
#include "ps/ps.h"

#ifndef lint
    static char afsid[]	= "$__Header$";
    static char rcsid[]	= "$Id$";
#endif

static char MARKERSIZE[] = "markersize";
static char LINEWIDTHSCALE[] = "lws";
static char CHAREXPANSION[] = "charexpansion";
static char CHARSPACING[] = "charspacing";
static char LINECOLOR[] = "lc";
static char MARKERCOLOR[] = "mc";
static char FILLCOLOR[] = "fc";
static char TEXTCOLOR[] = "tc";
static char header[] = 
#include "ps/headerv4.0.h"
;
static Gint	postscript_version = 1;

/* 11 bytes dummy for this implementation */
static Gchar	dummy[] = "dummy info.";

/* String array for formats created on the fly */
static Gchar	fmt[80];

static Gtxalign PSalign;
static Gpoint PSup, PSbase;

static Gflinter FillStyle = GHOLLOW;

/* Error messages are switched on and off through the environment */
/* variable XGKS_LOG anded with the enum MsgLog */

typedef enum MsgLog {
  INFO = 1,
  WARN = 2,
  ERR = 4
} MsgLog;

static int LogFlag = ERR;

static void initLogFlag()
{
  char *cp = getenv("XGKS_LOG");
  if (cp){
    LogFlag = atoi(cp);
  }
}

static void msgInfo(char *format, ...)
{
  if (LogFlag & INFO){
    va_list ap;
    va_start(ap, format);
    /* print out name of function causing error */
    fprintf(stderr, "XGKS(PS): Info: ");
    /* print out remainder of message */
    vfprintf(stderr, format, ap);
    va_end(ap);
  }
}

static void msgWarn(char *format, ...)
{
  if (LogFlag & WARN){
    va_list ap;
    va_start(ap, format);
    /* print out name of function causing error */
    fprintf(stderr, "XGKS(PS): Warning: ");
    /* print out remainder of message */
    vfprintf(stderr, format, ap);
    va_end(ap);
  }
}

static void msgErr(char *format, ...)
{
  if (LogFlag & ERR){
    va_list ap;
    va_start(ap, format);
    /* print out name of function causing error */
    fprintf(stderr, "XGKS(PS): Error: ");
    /* print out remainder of message */
    vfprintf(stderr, format, ap);
    va_end(ap);
  }
}

static char *LineStyles[] = {
    "",
    "[] 0 sd",
    "[5 4] 0 sd",
    "[3] 0 sd",
    "[4 2 2 2] 0 sd",
    0
};

void allocate_color(mf_cgmo *cgmo, int idx, float r, float g, float b)
{
  /* Gray value from PS 2.0 Red book, p. 304 */
  float gray = .3 * r + .59 * g + .11 * b;
  FILE *fp = cgmo->fp;
  msgInfo("allocate_color: assigning color (%f, %f, %f) to %d\n",
	  r, g, b, idx);
  (void) fprintf(fp, "ct %d [%f %f %f] put\n", idx, r, g, b);
  (void) fprintf(fp, "nct %d [%d %d %d] put\n", idx,
		 (int)(255.*r), (int)(255.*g), (int)(255.*b));
  (void) fprintf(fp, "gct %d %f put\n", idx, gray);
}

static void set_lineStyle(mf_cgmo *cgmo, Gint attr, Gasf type)
{
  if (type != xgks_state.gks_lnattr.type)
    return;
  if (type == GBUNDLED){
    attr = cgmo->ws->lnbundl_table[attr].type;
  }
  msgInfo("set_lineStyle: setting style to %d\n", attr);
  if (attr <= 0 || attr > 4)
    attr = 1;
  fprintf(cgmo->fp, "%s\n", LineStyles[attr]);
}

static void
set_lineWidth(mf_cgmo *cgmo, double size, Gint attr, Gasf type)
{
  if (type != xgks_state.gks_lnattr.width)
    return;
  if (type == GBUNDLED){
    size = cgmo->ws->lnbundl_table[attr].width;
  }
  msgInfo("set_lineWidth: setting width to %lf\n", size);
  if (size < 1.0)
    size = 1.0;
  size *= 0.5;
  fprintf(cgmo->fp, "/%s %.6lf def\n", LINEWIDTHSCALE, size);
}

static void set_lineColor(mf_cgmo *cgmo, Gint attr, Gasf type)
{
  if (type != xgks_state.gks_lnattr.colour)
    return;
  if (type == GBUNDLED){
    attr = cgmo->ws->lnbundl_table[attr].colour;
  }
  msgInfo("set_lineColor: setting color index to %d\n", attr);
  fprintf(cgmo->fp, "/%s %d def\n", LINECOLOR, attr);
  
}


/*
 * Return a string identifying the user and installation.
 */
    static Gchar*
XgksMAuthor(void)
{
    char		*username	= getlogin();
    struct utsname	name;
    static Gchar	buffer[41];

    buffer[0]	= 0;

    if (username != NULL)
	(void) strncat(buffer, username, sizeof(buffer) - 1);

    if (uname(&name) != -1) {
	int	nchr	= strlen(buffer);

	if (nchr < sizeof(buffer) - 1) {
	    buffer[nchr++]	= '@';
	    (void) strncpy(buffer + nchr, name.nodename, 
			   sizeof(buffer) - nchr - 1);
	}
    }

    return buffer;
}


/*
 * Return a date-string.
 */
    static Gchar*
XgksMDate(void)
{
    time_t          clock = time((time_t *) NULL);
    static Gchar    date[9];

    (void) strftime(date, (size_t)sizeof(date), "%y/%m/%d", 
                    localtime(&clock));

    return date;
}


/*
 * Open an output PostScript file.
 */
    int
PSmoOpen(WS_STATE_PTR ws)
{
  Gint status = 1;
    initLogFlag();
    assert(ws != NULL);
    if ((ws->mf.cgmo = (mf_cgmo*)umalloc(sizeof(mf_cgmo))) != NULL) {
	mf_cgmo	*cgmo	= ws->mf.cgmo;

	if ((cgmo->fp = fopen(ws->conn, "w")) == NULL) {
	    (void)PSmoClose(&ws->mf);
	} else {
	    cgmo->ws			= ws;
	    cgmo->type			= MF_PS;
	    fputs(header, ws->mf.any->fp);
	    fputs("\n%%Page:1 1\n", ws->mf.any->fp);
	    allocate_color(cgmo, 0, 1, 1, 1);
	    allocate_color(cgmo, 1, 0, 0, 0);
	    status	= OK;
	}
    }

    return status;
}


/* True hack here -- if aspect is > ratio of page height to page length */
/* than use portrait, otherwise landscape. This assumes 8.5x11 */
/* We are screwed if paper dimensions are different (i.e. A4) */

void PSresize(WS_STATE_PTR ws, Gpoint size)
{
  float myAspect = size.y/size.x;
  FILE	*fp	= ws->mf.any->fp;
  if (myAspect < 1.0){
    PSmoSetLandscape(&ws->mf);
  }
  fprintf(fp, "gr %f center gs\n", myAspect);
  
}

/*
 * Special function called from gescsetlandscape to set landscape mode
 */

int PSmoSetLandscape(Metafile *mf)
{
    FILE	*fp	= mf->any->fp;
    fputs("gr pagewidth neg Landscape gs\n", fp);
    return OK;
}

/*
 * Close an output PostScript file.
 */
    int
PSmoClose(Metafile *mf)
{
  int status = 1;		/* return status error */
  if (mf != NULL && mf->cgmo != NULL) {
    mf_cgmo *cgmo	= mf->cgmo;
    FILE *fp = cgmo->fp;

    if (fp != NULL) {
      fputs("showpage\n", fp);
      fputs("%%Trailer\n", fp);
      fputs("%%Pages: 1\n", fp);
      fputs("frpagesave restore\n", fp);
      if (!ferror(fp) & fclose(fp) != EOF)
	status	= OK;
    }

    ufree((voidp)mf->cgmo);
    mf->cgmo	= NULL;
  }

  return status;
}


/*
 * Set the clear flag in an output PostScript file.
 */
    int
PSclear(Metafile *mf, int num, Gclrflag flag)
{
  if (mf != NULL && mf->cgmo != NULL) {
    mf_cgmo *cgmo	= mf->cgmo;
    FILE *fp = cgmo->fp;
    if (fp != NULL){
      fprintf(fp, "gr gs clippath 1 setgray fill\n");
    }
  }
  return OK;
}


/*
 * Redraw all segments in an output PostScript file.
 */
    int
PSredrawAllSeg(Metafile **mf, int num)
{
				/* Noop */
  msgWarn("PSredrawAllSeg: Don't support this feature\n");
  return OK;
}


/*
 * Set the update flag in an output PostScript file.
 */
     int
PSupdate(Metafile **mf, int num, Gregen regenflag)
{
				/* Noop */
  msgWarn("PSupdate: Don't support this feature\n");
  return OK;
}


/*
 * Set the deferal state in an output PostScript file.
 */
    int
PSdefer(Metafile **mf, int num, Gdefmode defer_mode, Girgmode regen_mode)
{
				/* Noop */
  msgWarn("PSdefer: Don't support this feature\n");
  return OK;
}


/*
 * Write a message to an output PostScript file.
 */
    int
PSmessage(Metafile **mf, int num, Gchar *string)
{
				/* Noop */
  msgWarn("PSmessage: Don't support this feature\n");
  return OK;
}




/*
 * Write text to an output PostScript file.
 */
    int
PStext(Metafile *mf, int num, Gpoint *at, Gchar *string)
{
  msgWarn("PStext: Don't support this feature\n");
  return OK;

#if 0
    int		imf;
    mf_cgmo		**cgmo	= &mf->cgmo;
    if (!at || !string || *string == 0)
      return OK;

    for (imf = 0; imf < num; ++imf) {
	Gint	i, length, count = 0;
	FILE	*fp	= cgmo[imf]->fp;
	Gpoint* up = &PSup;
	char *nstr;

				/* Have to escape '(' , ')' and '\' charcters */
	length = strlen(string);
	nstr = malloc(2 * length);
	for (i=0; i < length; i++){
	    if (string[i] == '\\' || string[i] == '(' || string[i] == ')'){
                nstr[count++] = '\\';
	    }
	    nstr[count++] = string[i];
	}
	nstr[count] = 0;

	fprintf(fp,"gs %s o %.6f %.6f m (%s) ", TEXTCOLOR, at->x, at->y, nstr);
	if (up->y != 1.0 && up->x != 0.0){
	    double angle = atan2(-up->x, up->y) * 180./M_PI;
	    fprintf(fp, "%.0lf rotate htext ", angle);
	}
	switch (PSalign.hor){
	  case GTH_RIGHT:
	    fputs("rtext ", fp);
	    break;
	  case GTH_CENTRE:
	    fputs("ctext ", fp);
	    break;
	default:
	  break;
	}
	switch (PSalign.ver){
	  case GTV_NORMAL:
	    break;
	  case GTV_TOP:
	    fputs("PStext: Vertical alignment GTV_TOP not supported\n", stderr);
	    break;
	  case GTV_CAP:
	    fputs("PStext: Vertical alignment GTV_CAP not supported\n", stderr);
	    break;
	  case GTV_HALF:
	    fputs("htext ", fp);
	    break;
	  case GTV_BASE:
	    fputs("PStext: Vertical alignment GTV_BASE not supported\n", stderr);
	    break;
	  case GTV_BOTTOM:
	    fputs("PStext: Vertical alignment GTV_BOTTOM not supported\n", stderr);
	    break;
	  default:
	    fputs("PStext: Vertical alignment UNKNOWN not supported\n", stderr);
	    break;
	}
	fputs("show gr\n", fp);
    }
#ifdef PSDEBUG
    fprintf(stderr, "PStext: text = %s at (%.6f %.6f)\n", str, at->x, at->y);
#endif
    return OK;
#endif /* #if 0 */
}

/*
 * Write a cell array to an output PostScript file.
 * New method fakes transparency by not writing image data for colors of 0
 */

    int
PScellArray(Metafile *mf, int num, Gpoint *ll, Gpoint *ur, Gpoint *lr, Gint row, Gint *colour, Gipoint *dim)
{
  msgWarn("PScellArray: Don't support this feature\n");
  return OK;
#if 0
    int imf;
    mf_cgmo		**cgmo	= &mf->cgmo;
    if (ll->x > ur->x){
	float tmp = ll->x;
	ll->x = ur->x;
	ur->x = tmp;
    }
    for (imf = 0; imf < num; ++imf) {
	FILE	*fp	= cgmo[imf]->fp;
	int j;

	for (j=0; j < dim->y; ++j){
	    int k = 0, start = -1, end = -1, currcolor = -1;
	    while (k < dim->x) {
		for (start = -1, k= end + 1; k < dim->x; ++k){
		    if (colour[j*dim->x + k] != 0){
			start = k;
			end = dim->x;
			currcolor = colour[j*dim->x + k];
			break;
		    }
		}
		for (k = k + 1; k < dim->x; ++k){
		    if (colour[j*dim->x + k] != currcolor){
			end =k;
			break;
		    }
		}
		if (start > -1){
		    float widthx = ur->x - ll->x;
		    float startx = ll->x + (float)start/(float)dim->x*widthx;
		    float endx = ll->x + (float)(end+1)/(float)dim->x*widthx;

		    float widthy = ur->y - ll->y;
		    float deltay = widthy / dim->y;

		    if (endx > ur->x)
		      endx = ur->x;

		    fprintf(fp, "%d o %.6f %.6f %.6f %.6f b f\n",
			    colour[j*dim->x + start],
			    startx, ll->y + j * deltay,
			    endx, ll->y + (j + 1) * deltay + 1e-6);

		}
	    }
	}
    }
    
    return OK;
#endif /* #if 0 */
}


/*
 * Close a segment in an output PostScript file.
 */
    int
PScloseSeg(Metafile *mf, int num)
{
				/* Noop */
  msgWarn("PScloseSeg: Don't support this feature\n");
  return OK;
}


/*
 * Set the graphic attributes in an output PostScript file.
 */

/*
 * Write a graphic to output PostScript files.
 *
 * This routine is suitable for
 *
 *	POLYLINE    -- code == 11
 *	POLYMARKER  -- code == 12
 *	FILLAREA    -- code == 14
 */
/* RETURN HERE ****************/

    int
PSoutputGraphic(Metafile *mf, int num, Gint code, Gint num_pt, Gpoint *pos)
{
    int		imf;
    mf_cgmo		**cgmo	= &mf->cgmo;

    for (imf = 0; imf < num; ++imf) {
	Gint	i;
	FILE	*fp	= cgmo[imf]->fp;

	assert(num_pt > 0);
	switch(code){
	  case GKSM_FILL_AREA:
	    if (FillStyle == GHOLLOW){
	      fprintf(fp, "0.5 w %s o %.6f %.6f m\n", 
		      FILLCOLOR, pos->x, pos->y);
	    } else {
	      fprintf(fp, "%s o %.6f %.6f m\n", FILLCOLOR, pos->x, pos->y);
	    }
	    break;
	  case GKSM_POLYLINE:
	    fprintf(fp, "%s w %s o %.6f %.6f m\n", LINEWIDTHSCALE,
		    LINECOLOR, pos->x, pos->y);
	    break;
	  case GKSM_POLYMARKER:
	    fprintf(fp, "%s o %.6f %.6f pm\n", MARKERCOLOR, pos->x, pos->y);
	    break;
	  default:
	    fprintf(stderr, "PSoutputGraphics: Unknown code %d\n", code);
	    return OK;
	}
	{
	  Gpoint *npos = pos;
	  npos++;
	  for (i = 1; i < num_pt; ++i,++npos) {
	    if (code == GKSM_POLYMARKER)
	      (void) fprintf(fp, "%.6f %.6f pm\n", npos->x, npos->y);
	    else
	      (void) fprintf(fp, "%.6f %.6f l\n", npos->x, npos->y);
	  }
	}
	switch(code){
	  case GKSM_FILL_AREA:
	    if (FillStyle == GHOLLOW){
	      fprintf(fp, "%s\n", LineStyles[1]);
	      fprintf(fp, "%.6f %.6f l\n", pos->x, pos->y);
	      fprintf(fp, "t\n");
	    } else {
	      fprintf(fp, "f\n");
	    }
	    break;
	  case GKSM_POLYMARKER:
	    fprintf(fp, "f\n");
	    break;
	  case GKSM_POLYLINE:
	    fprintf(fp, "t\n");
	    break;
	}
    }
    return OK;
}
/*
 * Set the size of graphics in an output PostScript file.
 */
    int
PSsetGraphSize(Metafile *mf, int num, Gint code, double size)
{
    int imf;
    mf_cgmo		**cgmo	= &mf->cgmo;

    for (imf = 0; imf < num; ++imf) {
      switch(code){
      case GKSM_LINEWIDTH_SCALE_FACTOR:
	set_lineWidth(cgmo[imf], size, 0, GINDIVIDUAL);
	break;
      case GKSM_CHARACTER_EXPANSION_FACTOR:
      case GKSM_MARKER_SIZE_SCALE_FACTOR:
      case GKSM_CHARACTER_SPACING:
	msgWarn("PSsetGraphSize: Don't support code %d\n", code);
	break;
      default:
	msgWarn("PSsetGraphSize: Unknown code %d\n", code);
      }
    }
    return OK;
}



int
PSsetGraphAttr(Metafile *mf, int num, Gint code, Gint attr)
{
    int		imf;
    mf_cgmo		**cgmo	= &mf->cgmo;
    char *comm = 0;
    for (imf = 0; imf < num; ++imf) {
      switch(code){
      case GKSM_POLYLINE_INDEX:
	set_lineStyle(cgmo[imf], attr, GBUNDLED);
	set_lineColor(cgmo[imf], attr, GBUNDLED);
	set_lineWidth(cgmo[imf], 0.0, attr, GBUNDLED);
	break;
      case GKSM_POLYLINE_COLOUR_INDEX:
	set_lineColor(cgmo[imf], attr, GINDIVIDUAL);
	break;
      case GKSM_LINETYPE:
	set_lineStyle(cgmo[imf], attr, GINDIVIDUAL);
	break;
      case GKSM_POLYMARKER_COLOUR_INDEX:
	comm = MARKERCOLOR;
	break;
      case GKSM_TEXT_COLOUR_INDEX:
	comm = TEXTCOLOR;
	break;
      case GKSM_FILL_AREA_COLOUR_INDEX:
	comm = FILLCOLOR;
	break;
      case GKSM_MARKER_TYPE:
      case GKSM_POLYMARKER_INDEX:
      case GKSM_FILL_AREA_INDEX:
      case GKSM_FILL_AREA_STYLE_INDEX:
      case GKSM_PICK_IDENTIFIER:
      case GKSM_TEXT_INDEX:
	msgWarn("PSsetGraphAttr: Don't support code %d\n", code);
	/* Ignore */
	break;
      default:
	msgWarn("PSsetGraphAttr: Unknown code %d\n", code);
      }

      if (comm)
	fprintf(cgmo[imf]->fp, "/%s %d def\n", comm, attr);
    }
    return OK;
}


/*
 * Set the font precision in an output PostScript file.
 */
    int
PSsetTextFP(Metafile *mf, int num, Gtxfp *txfp)
{
    if (!txfp)
      return OK;
    msgWarn("PSsetTextFp: Don't support this feature\n");
    return OK;
}

#define HYPOT(x,y) sqrt((double)((x)*(x) + (y)*(y)))


/*
 * Set the character up-vector and character height in an output PostScript file.
 */
    int
PSsetCharUp(Metafile *mf, int num, Gpoint *up, Gpoint *base)
{
    msgWarn("PSsetCharUp: Don't support this feature\n");
    return OK;
#if 0
    float height;
    int imf;
    mf_cgmo		**cgmo	= &mf->cgmo;
    if (up == 0 || base == 0)
      return OK;

    PSup = *up;
    PSbase = *base;
    height = HYPOT(up->x, up->y);

    for (imf = 0; imf < num; ++imf) {
	FILE	*fp	= cgmo[imf]->fp;
	fprintf(fp, "%.6f sf\n", height);
    }

#ifdef PSDEBUG
    fprintf(stderr, "PSsetCharUp: up=(%.6f %.6f) base=(%.6f %.6f) height = %.6f\n",
	    up->x, up->y,
	    base->x, base->y, height);
#endif
    return OK;
#endif /* #if 0 */
}


/*
 * Set the text-path in an output PostScript file.
 */
    int
PSsetTextPath(Metafile *mf, int num, Gtxpath path)
{
    msgWarn("PSsetTextPath: Don't support this feature\n");
    return OK;
}


/*
 * Set the text-alignment in an output PostScript file.
 */
    int
PSsetTextAlign(Metafile *mf, int num, Gtxalign *align)
{
    msgWarn("PSsetTextAlign: Don't support this feature\n");
    return OK;
#if 0
    if (align == 0)
      return;
    PSalign = *align;
#ifdef PSDEBUG
    fprintf(stderr, "PSsetTextAlign: align = %d %d\n", align->hor, align->ver);
#endif
    return OK;
#endif /* #if 0 */
}


/*
 * Set the interior fill-style in an output PostScript file.
 */
    int
PSsetFillStyle(Metafile *mf, int num, Gflinter style)
{
  FillStyle = style;
  return OK;
}


/*
 * Set the pattern size in an output PostScript file.
 */
    int
PSsetPatSize(Metafile *mf, int num)
{
    msgWarn("PSsetPatSize: Don't support this feature\n");
    return OK;
}


/*
 * Set the pattern reference-point in an output PostScript file.
 */
    int
PSsetPatRefpt(Metafile *mf, int num)
{
    msgWarn("PSsetPatRefpt: Don't support this feature\n");
    return OK;
}


/*
 * Set the ASF in an output PostScript file.
 */
    int
PSsetAsf(Metafile *mf, int num)
{
    msgWarn("PSsetAsf: Don't support this feature\n");
    return OK;
}


/*
 * Set the line and marker representation in an output PostScript file.
 */
    int
PSsetLineMarkRep(Metafile *mf, int num, Gint code, Gint idx, Gint type, double size, Gint colour)
{
    msgWarn("PSsetLineMarkRep: Don't support this feature\n");
    return OK;
}


/*
 * Set the text representation in an output PostScript file.
 */
    int
PSsetTextRep(Metafile *mf, int num, Gint idx, Gtxbundl *rep)
{
    msgWarn("PSsetTextRep: Don't support this feature\n");
    return OK;
#if 0
    if (!rep)
      return;
#ifdef PSDEBUG
    fprintf(stderr, "PSsetTextRep: index %d font prec (%d %d) exp %.6f sp %.6f color %d\n",
	    idx, rep->fp.font, rep->fp.prec,
	    rep->ch_exp, rep->space, rep->colour);
#endif
    return OK;
#endif
}


/*
 * Set the fill representation in an output PostScript file.  
 */
    int
PSsetFillRep(Metafile *mf, int num, Gint idx, Gflbundl *rep)
{
    msgWarn("PSsetFillRep: Don't support this feature\n");
    return OK;
}


/*
 * Set the pattern representation in an output PostScript file.
 */
    int
PSsetPatRep(Metafile *mf, int num, Gint idx, Gptbundl *rep)
{
    msgWarn("PSsetPatRep: Don't support this feature\n");
    return OK;
}


/*
 * Set the colour representation in an output PostScript file.
 * Also write out gray scale version
 */
#ifndef MAX
#define MAX(a,b)    (((a)>(b))?(a):(b))
#endif

    int
PSsetColRep(Metafile *mf, int num, Gint idx, Gcobundl *rep)
{
    int		imf;
    mf_cgmo		**cgmo	= &mf->cgmo;

    for (imf = 0; imf < num; ++imf) {
	FILE	*fp	= cgmo[imf]->fp;
	float r = rep->red, g = rep->green, b = rep->blue;
	allocate_color(cgmo[imf], idx, r, g, b);
    }

    return OK;
}


/*
 * Set the clipping rectangle in an output PostScript file.
 * Note that an unpaired grestore
 * is used to restore the initial clipping path before setting up the
 * clip; this means that no other unpaired gsaves may be used.
 */
    int
PSsetClip(Metafile *mf, int num, Glimit *rect)
{
    int		imf;

    mf_cgmo		**cgmo	= &mf->cgmo;
    for (imf = 0; imf < num; ++imf) {
      FILE	*fp	= cgmo[imf]->fp;
      
      (void) fprintf(fp, "gr gs %f %f %f %f b clip n\n", 
		     rect->xmin, rect->ymin,
		     rect->xmax, rect->ymax);
    }
    return OK;
}


/*
 * Set the viewport limits in an output PostScript file.
 */
    int
PSsetLimit(Metafile *mf, int num, Gint code, Glimit *rect)
{
    msgWarn("PSsetLimit: Don't support this feature\n");
    return OK;
}


/*
 * Rename a segment in an output PostScript file.
 */
    int
PSrenameSeg(Metafile *mf, int num, Gint old, Gint new)
{
    msgWarn("PSrenameSeg: Don't support this feature\n");
    return OK;
}


/*
 * Set the segment transformation in an output PostScript file.
 */
    int
PSsetSegTran(Metafile *mf, int num, Gint name, Gfloat (*matrix)[])
{
    msgWarn("PSsetSegTran: Don't support this feature\n");
    return OK;
}


/*
 * Set the segment attributes in an output PostScript file.
 */
    int
PSsetSegAttr(Metafile *mf, int num, Gint name, Gint code, Gint attr)
{
    msgWarn("PSSetSegAttr: Don't support this feature\n");
    return OK;
}


/*
 * Set the segment visibility in an output Metafile.
 */
    int
PSsetSegVis(Metafile *mf, int num, Gint name, Gsegvis vis)
{
    msgWarn("PSsetSegVis: Don't support this feature\n");
    return OK;
}


/*
 * Set segment highlighting in an output PostScript file.
 */
    int
PSsetSegHilight(Metafile *mf, int num, Gint name, Gseghi hilight)
{
    msgWarn("PSsetSegHiglight: Don't support this feature\n");
    return OK;
}


/*
 * Set segment priority in an output PostScript file.
 */
    int
PSsetSegPri(Metafile *mf, int num, Gint name, double pri)
{
    msgWarn("PSsetSegPri: Don't support this feature\n");
    return OK;
}


/*
 * Set segment detectability in an output PostScript file.
 */
    int
PSsetSegDetect(Metafile *mf, int num, Gint name, Gsegdet det)
{
    msgWarn("PSsetSegDetect: Don't support this feature\n");
    return OK;
}
