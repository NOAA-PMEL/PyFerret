/*
*
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

* copy the contents of the active plotting window to the double buffered
* window for animation purposes

* */


#include "gks_implem.h"
#include "wslist.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>

#ifdef NO_ENTRY_NAME_UNDERSCORES
copy_buffered_window(ws_id, anim_id)
#else
copy_buffered_window_(ws_id, anim_id)
#endif
     int *ws_id, *anim_id;

{
  WS_STATE_ENTRY *ws, *anim, *temp_win;
  unsigned width, height;
  XWindowAttributes win_info, anim_win_info;
  int src_absx, src_absy, dest_absx, dest_absy;
  int screen, tmp_id;
  Window dummywin;
  Pixmap pixmap;

/* determine the XGKS ws state entry structure from ws_id */
  ws  = OPEN_WSID (*ws_id);
  anim = OPEN_WSID(*anim_id);
  

  /*
   * Get the parameters of the window being dumped.
   */
  if(!XGetWindowAttributes(ws->dpy, ws->win, &win_info)) 
    {
      fprintf (stderr, "Can't get target window attributes.");
      exit(1);
    }
  
  screen = DefaultScreen(ws->dpy);
  
  /* handle any frame window */
  if (!XTranslateCoordinates (ws->dpy, ws->win, 
			      RootWindow (ws->dpy, screen), 0, 0, 
			      &src_absx, &src_absy, &dummywin)) {
    fprintf (stderr,  "unable to translate window coordinates\n");
    }  
   
  
  /*
   * Get the parameters of the window being dumped to.
   */
  if(!XGetWindowAttributes(anim->dpy, anim->win, &anim_win_info)) 
    {
      fprintf (stderr, "Can't get target window attributes.");
      exit(1);
    }

  screen = DefaultScreen(anim->dpy);
  
  /* handle any frame window  */
  if (!XTranslateCoordinates (ws->dpy, anim->win, 
			      RootWindow (anim->dpy, screen), 
			      0, 0, &dest_absx, &dest_absy, &dummywin)) {
    fprintf (stderr, "unable to translate window coordinates (%d,%d)\n");
  } 

  width = win_info.width;
  height = win_info.height;

  /*    XCopyArea(ws->dpy, ws->win, anim->win, DefaultGC(ws->dpy, 
	    DefaultScreen(ws->dpy)), src_x, src_y, 
            width, height, (anim_win_info.x), (anim_win_info.y-30));  */

  XCopyArea(ws->dpy, ws->win, anim->win, DefaultGC(ws->dpy,
	    DefaultScreen(ws->dpy)), 0, 0,
	    width, height, 0,0);  
  XSync(ws->dpy, 0);
  /*	xgks_x_events_(); */


}

