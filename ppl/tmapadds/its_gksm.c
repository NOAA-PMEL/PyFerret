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



/* its_gksm.c

* boolean function returns 1 or 0 to tell if the given
* GKS workstation is a GKSM metafile workstation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 4/9/97

* compile with
*    cc -g -c -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib  -I$TMAP_LOCAL/src/xgks-2.5.5/port/ its_gksm.c
*  or
*    cc    -c -I$TMAP_LOCAL/src/xgks-2.5.5/src/lib  -I$TMAP_LOCAL/src/xgks-2.5.5/port/ its_gksm.c

*/

/* include files from XGKS */
#include "udposix.h"
#include "gks_implem.h"

/* its_gksm */
#ifdef NO_ENTRY_NAME_UNDERSCORES
int its_gksm(int *wkid)
#else
int its_gksm_(int *wkid )
#endif
{
  WS_STATE_PTR    ws;
  ws = OPEN_WSID((Gint) *wkid);

  if (ws->ewstype == MO){
/* yes - a metafile workstation. Is it GKSM? */
    if (ws->mf.any->type     == MF_GKSM)
      return(1);
    else
      return(0);
  }
  else
/* no - not even a metafile workstation */
    return(0);
}


      


