/* Resize X window in XGKS, change aspect if req'd. 
 * -> Size values passed should be in meters.
 * J Davison 8.24.93/12.17.93
 *
 * Mod to use in X preview of metafiles *jd* 8.23.95 Convert inches to metres
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 
#include "udposix.h"
#include "gks_implem.h"

resize_xgks_window (ws_id, x, y)

Gint ws_id;
float x;
float y;

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

/***************************************************************************/

/* Convert x,y from inches to meters */
  x *= 0.0254;
  y *= 0.0254;

  ws  = OPEN_WSID (ws_id);
  scr = DefaultScreen (ws->dpy);

  xf = ((float) DisplayWidth(ws->dpy,scr)) /((float) DisplayWidthMM(ws->dpy,scr));
  yf = ((float) DisplayHeight(ws->dpy,scr))/((float) DisplayHeightMM(ws->dpy,scr));
  ix = x*1000.0*xf;
  iy = y*1000.0*yf;

  if (x > y) {
    aspect = y/x;
    size.x = 1000;
    size.y = 1000 * aspect;
    }
  else {
    aspect = x/y;
    size.x = 1000 * aspect;
    size.y = 1000;
    }
  gescsetdcsize (ws_id, size);

  XResizeWindow (ws->dpy,ws->win,ix,iy);

  tp = &t_now;
  t0 = time(0);

  do { 
       xw_event = XCheckWindowEvent (ws->dpy,ws->win,StructureNotifyMask,&evnt);
       time (tp);
     } while (xw_event && (t_now - t0 < 3));

}





