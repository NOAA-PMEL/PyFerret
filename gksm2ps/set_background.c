/* Set background color in XGKS. 
 * Values passed are ws id and color index.
 * J Davison 1.11.94
 *
 * *jd* Hacked for metafile interpretation 8.24.95
 */

#include "udposix.h"
#include <stdlib.h>
#include <string.h>
#include <gks_implem.h>

set_background (ws_id, ndx)
Gint ws_id; 
int ndx;

{
  WS_STATE_ENTRY *ws;

  Display **dpy;
  Window   *win;
  GC        *gc;

  Gint            val;
  int             scr;
  int             stat;

/****************************************************************************/

  ws  = OPEN_WSID (ws_id);
  scr = DefaultScreen (ws->dpy);

  if (ndx == 0)
    XSetWindowBackground (ws->dpy,ws->win,BlackPixel(ws->dpy,scr));
  else
    XSetWindowBackground (ws->dpy,ws->win,WhitePixel(ws->dpy,scr));
}





