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



/* Resize X window in XGKS, change aspect if req'd.  Callable from FORTRAN.  
 * -> Size values passed are in meters.
 * J Davison 8.24.93/12.17.93
 */

/* Changed include order of gks_implem.h to remove errors in compile (set 
 * **before** stdlib.h) for linux port *jd* 1.28.97
 * More hacks to support batch mode (i.e. no display available) *js* 8.97
 * Prevent null pointer dereference for ws if no X display available *js* 2.99
 */

/* Remove XCheckWindowEvent to see if that's causing the occasional hangs
 * *jd* 2.23.99
 */

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include "udposix.h"
#include "gks_implem.h"
#include "cgm/cgm.h"
#include "cgm/cgm_implem.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 

#ifdef NO_ENTRY_NAME_UNDERSCORES
resize_xgks_window (ws_id, x, y)
#else
resize_xgks_window_ (ws_id, x, y)
#endif

Gint *ws_id;
float *x;
float *y;

{
  WS_STATE_ENTRY *ws;

  Display       **dpy;
  Window         *win;
  GC             *gc;

  XEvent          evnt;
  Gint            val;
  Gpoint          size;

  unsigned int    ix,iy;
  float           xf,yf,aspect;

  int             xw_event,scr; 
  time_t          t0,t_now,*tp;

/*****************************************************************************/

  ws  = OPEN_WSID (*ws_id);
  if (ws && ws->dpy){
    scr = DefaultScreen (ws->dpy);

    xf = ((float) DisplayWidth(ws->dpy,scr)) /((float) DisplayWidthMM(ws->dpy,scr));
    yf = ((float) DisplayHeight(ws->dpy,scr))/((float) DisplayHeightMM(ws->dpy,scr));
  } else {
    xf = 1280.0/361.0;		/* Standard 20" monitor width/size */
    yf = 1024.0/289.0;
  }

  ix = (*x)*1000.0*xf;
  iy = (*y)*1000.0*yf;

  if (*x > *y) {
    aspect = (*y / *x);
    size.x = 1000;
    size.y = 1000 * aspect;
    }
  else {
    aspect = (*x / *y);
    size.x = 1000 * aspect;
    size.y = 1000;
    }
  gescsetdcsize (*ws_id, size);

  if (ws){
    if (ws->ewstype == X_WIN && ws->dpy){
      XResizeWindow (ws->dpy,ws->win,ix,iy);
      tp = &t_now;
      t0 = time(0);
  
/*
 *    do { 
 *    xw_event = XCheckWindowEvent (ws->dpy,ws->win,StructureNotifyMask,&evnt);
 *     time (tp);
 *   } while (xw_event && (t_now - t0 < 3));
 *
 */

    } else if (ws->ewstype == MO){
      int type = ws->mf.cgmo->type;
      if (type == MF_GIF){
	Gpoint nsize;
	nsize.x = ix;
	nsize.y = iy;
	GIFresize(ws, nsize);
      } else if (type == MF_PS){
	PSresize(ws, size);
      }
    }
  }
}
