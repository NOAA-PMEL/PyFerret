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
resize_animation_window (ws_id, x, y)
#else
resize_animation_window_ (ws_id, x, y)
#endif

Gint *ws_id;
float *x;
float *y;

{
  WS_STATE_ENTRY *ws;

  ws  = OPEN_WSID (*ws_id);

  /* 3/02 *kob* need to cast x and y for IRIX */
  XResizeWindow (ws->dpy,ws->win,(unsigned int)x,(unsigned int)y);

 
}
