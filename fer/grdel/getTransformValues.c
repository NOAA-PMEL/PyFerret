/*
 * Drawing commands
 */
#include <Python.h> /* make sure Python.h is first */
#include "grdel.h"
#include "cferbind.h"

/*
 * Assigns the transformation values my, sx, sy, dx, and dy used
 * to convert user coordinate (userx, usery) to device coordinate
 * (devx, devy) using the formulae:
 *    devx = userx * sx + dx
 *    devy = (my - usery) * sy + dy
 */
void grdelGetTransformValues(double *my, double *sx, double *sy,
                                         double *dx, double *dy)
{
   float lftfrc, rgtfrc, btmfrc, topfrc;
   float lftcrd, rgtcrd, btmcrd, topcrd;
   float winwidth, winheight;
   double devlft, devtop, devwidth, devheight;
   double usrlft, usrtop, usrwidth, usrheight;

   FORTRAN(fgd_get_view_limits)(&lftfrc, &rgtfrc, &btmfrc, &topfrc,
                        &lftcrd, &rgtcrd, &btmcrd, &topcrd);
   FORTRAN(fgd_get_window_size)(&winwidth, &winheight);

   devlft     = (double) lftfrc * (double) winwidth;
   devwidth   = (double) rgtfrc * (double) winwidth;
   devwidth  -= devlft;
   devtop     = (1.0 - (double) topfrc) * (double) winheight;
   devheight  = (1.0 - (double) btmfrc) * (double) winheight;
   devheight -= devtop;

   usrlft = (double) lftcrd;
   usrwidth = (double) rgtcrd - usrlft;
   usrtop = 0.0;
   usrheight = (double) topcrd - (double) btmcrd;

   *my = (double) topcrd;
   *sx = devwidth / usrwidth;
   *sy = devheight / usrheight;
   *dx = devlft - (*sx) * usrlft;
   *dy = devtop - (*sy) * usrtop;
}

