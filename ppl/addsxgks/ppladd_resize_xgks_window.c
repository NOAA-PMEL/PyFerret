/* Resize X window in XGKS, change aspect if req'd.  Callable from FORTRAN.  
 * -> Size values passed are in meters.
 * J Davison 8.24.93/12.17.93
 */

/* Changed include order of gks_implem.h to remove errors in compile (set 
 * **before** stdlib.h) for linux port *jd* 1.28.97
 * More hacks to support batch mode (i.e. no display available) *js* 8.97
 */


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
  if (ws->dpy){
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

  if (ws->ewstype == X_WIN && ws->dpy){
    XResizeWindow (ws->dpy,ws->win,ix,iy);
    tp = &t_now;
    t0 = time(0);
  
    do { 
      xw_event = XCheckWindowEvent (ws->dpy,ws->win,StructureNotifyMask,&evnt);
      time (tp);
    } while (xw_event && (t_now - t0 < 3));
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






