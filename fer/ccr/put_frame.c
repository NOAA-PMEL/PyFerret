/* put_frame( ws_id, filename, status )
* dump an XGKS window as a GIF file

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 8/03/94
* 
* *kob* 5/25/95 replaced defunct ifdef confition AIX_XLF with current 
                NO_ENTRY_NAME_UNDERSCORES
compile with these flags to locate the include files:
        -I$TMAP_LOCAL/src/xgks-2.5.5/port \
        -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib \
        -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib/gksm

and optionally (non-ANSI cc compilers) with    -DNO_CC_PROTOTYPES
* *js* 9.97 added put_frame_batch 

*/


#include "gks_implem.h" /* ditto */
#include "wslist.h"
#include "cgm/cgm.h"		/* for public, API details */
#include "cgm/cgm_implem.h"		/* for implementation details */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>

#ifdef NO_ENTRY_NAME_UNDERSCORES
put_frame( ws_id, filename, errstr, format, status )
#else
put_frame_( ws_id, filename, errstr, format, status )
#endif
   char *filename, *errstr, *format;
   int *ws_id, *status;

{
  WS_STATE_ENTRY *ws;
  Display *mydisplay;
  Window   mywindow;

/* determine the XGKS ws state entry structure from ws_id */
  ws  = OPEN_WSID (*ws_id);	

/* the next 2 lines are diagnostic
     mydisplay = ws->dpy;
     mywindow  = ws->win;
*/

/* call up the capture routine */
/* Errors internal to Window_Dump com out in errstr */
   Window_Dump(ws->win,ws->dpy,filename,format);
   *status = 0;   /* not much use, really */

   return;
}

#ifdef NO_ENTRY_NAME_UNDERSCORES
void put_frame_batch(int *ws_id, char *filename, char *format,
			       char *errmsg, int *status)
#else
void put_frame_batch_(int *ws_id, char *filename, char *format,
			       char *errmsg, int *status)
#endif
{
  char oldfilename[BUFSIZ];
  WS_STATE_ENTRY *ws = OPEN_WSID(*ws_id);
  *status = 0;

  if (ws->mf.any->type != MF_GIF){
    strcpy(errmsg, "Batch FRAME only works for GIF files");
    return;
  }
  if (GIFFlush(&ws->mf, filename) != OK){
    sprintf(errmsg, "Couldn't write out GIF file %s\n", filename);
    return;
  }
}
      


