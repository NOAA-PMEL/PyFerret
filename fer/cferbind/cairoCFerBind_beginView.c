/* Python.h should always be first */
#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"

/*
 * Starts a "View" for this "Window".
 * In this case (Cairo), assigns the clipping rectangle
 * that may be used for the subsequent drawings.
 *
 * Arguments:
 *     lftfrac - view left edge as a fraction [0.0, 1.0) of the window size
 *     btmfrac - view bottom edge as a fraction (0.0, 1.0] of the window size
 *     rgtfrac - view right edge as a fraction (0.0, 1.0] of the window size
 *     topfrac - view top edge as a fraction [0.0, 1.0) of the window size
 *     clipit  - value passed to cairoCFerBind_clipView
 *               (clip drawing to this view rectangle?)
 *
 * The fractions are from the top left corner; thus lftfrac must be less than
 * rgtfrac and topfrac must be less than btmfrac.
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool cairoCFerBind_beginView(CFerBind *self, double lftfrac, double btmfrac,
                                  double rgtfrac, double topfrac, int clipit)
{
    CairoCFerBindData *instdata;
    int result;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_beginView: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }
    instdata = (CairoCFerBindData *) self->instancedata;

    /* Verify valid view fractions */
    if ( (0.0 > lftfrac) || (lftfrac >= rgtfrac) || (rgtfrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_beginView: invalid left (%#.3f) "
                             "and/or right (%#.3f) fractions", lftfrac, rgtfrac);
        return 0;
    }
    if ( (0.0 > topfrac) || (topfrac >= btmfrac) || (btmfrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_beginView: invalid top (%#.3f) "
                             "and/or bottom (%#.3f) fractions", topfrac, btmfrac);
        return 0;
    }

    /* Assign the view rectangle fractions */
    instdata->fracsides.left = lftfrac;
    instdata->fracsides.bottom = btmfrac;
    instdata->fracsides.right = rgtfrac;
    instdata->fracsides.top = topfrac;

    /* Assign clipping */
    result = self->clipView(self, clipit);

    return result;
}

