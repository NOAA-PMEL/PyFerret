/* Set background color in XGKS.  Callable from FORTRAN.  
 * Values passed are ws id and color index.
 * J Davison 1.11.94
 */

/* Changed include order of gks_implem.h to remove errors in compile (set 
 * **before** stdlib.h) for linux port *jd* 1.28.97
 */

#include "udposix.h"
#include "gks_implem.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef NO_ENTRY_NAME_UNDERSCORES
set_background (ws_id, ndx)
#else
set_background_ (ws_id, ndx)
#endif
Gint *ws_id; 
int *ndx;

{
  WS_STATE_ENTRY *ws;

  Display **dpy;
  Window   *win;
  GC        *gc;

  Gint            val;
  int             scr;
  int             stat;

/****************************************************************************/

  ws  = OPEN_WSID (*ws_id);
  scr = DefaultScreen (ws->dpy);

  if (*ndx == 0)
    XSetWindowBackground (ws->dpy,ws->win,BlackPixel(ws->dpy,scr));
  else
    XSetWindowBackground (ws->dpy,ws->win,WhitePixel(ws->dpy,scr));
}





