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


      


