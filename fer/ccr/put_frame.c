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
#include "ferret.h"

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

void FORTRAN(put_frame_batch)(int *ws_id, char *filename, char *format,
			       char *errmsg, int *status)
{
  char oldfilename[BUFSIZ];
  WS_STATE_ENTRY *ws = OPEN_WSID(*ws_id);
  *status = 0;

  if (ws == 0 || ws->mf.any == 0){
    strcpy(errmsg, "No open workstations for batch FRAME command");
    return;
  }

  if (ws->mf.any->type != MF_GIF){
    strcpy(errmsg, "Batch FRAME only works for GIF files");
    return;
  }
  if (GIFFlush(&ws->mf, filename) != OK){
    sprintf(errmsg, "Couldn't write out GIF file %s\n", filename);
    return;
  }
}
      
void FORTRAN(put_temp_frame_batch)(int *ws_id, char *filename, int *length)
{
  char format[BUFSIZ], errmsg[BUFSIZ];
  int status;
  char *tname = tempnam("/tmp", "fer");
  WS_STATE_ENTRY *ws = OPEN_WSID(*ws_id);
  status = 0;
  strcpy(filename, tname);
  strcat(filename, ".gif");
  FORTRAN(put_frame_batch)(ws_id, filename, format, errmsg, &status);
  *length = strlen(filename);
  free(tname);
}
      


