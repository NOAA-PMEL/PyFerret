/* Wait for signal of resize if resize happens.  Callable from FORTRAN.  
 * 
 * J Davison 3.7.94
 */

/* Changed include order of gks_implem.h to remove errors in compile (set 
 * **before** stdlib.h) for linux port *jd* 1.28.97
 */

#include "udposix.h"
#include "gks_implem.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 

#ifdef NO_ENTRY_NAME_UNDERSCORES
wait_on_resize (ws_id)
#else
wait_on_resize_ (ws_id)
#endif

Gint *ws_id;

{
  WS_STATE_ENTRY *ws;

  Display       **dpy;
  Window         *win;
  GC             *gc;

  XEvent          evnt;

  int             xw_event,scr; 
  time_t          t0,t_now,*tp;

/*****************************************************************************/

  ws  = OPEN_WSID (*ws_id);
  scr = DefaultScreen (ws->dpy);

  tp = &t_now;
  t0 = time(0);
  do { 
       xw_event = XCheckWindowEvent (ws->dpy,ws->win,StructureNotifyMask,&evnt);
       time (tp);
     } while (xw_event && (t_now - t0 < 3));
}

