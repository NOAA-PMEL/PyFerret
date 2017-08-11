/* Python.h should always be first */
#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "cairoCFerBind.h"
#include "FerMem.h"

/* Instantiate the global value */
const char *CCFBColorId = "CCFBColorId";

/*
 * Create a color object for this "Window".
 *
 * Arguments:
 *     redfrac    - red fraction [0.0, 1.0]
 *     greenfrac  - green fraction [0.0, 1.0]
 *     bluefrac   - blue fraction [0.0, 1.0]
 *     opaquefrac - opaque fraction [0.0, 1.0]
 *                  (0.0 is transparent, 1.0 is opaque)
 *
 * Returns a color object if successful.   If an error occurs,
 * grdelerrmsg is assigned an appropriate error message and NULL
 * is returned.
 */
grdelType cairoCFerBind_createColor(CFerBind *self, double redfrac,
                        double greenfrac, double bluefrac, double opaquefrac)
{
    CCFBColor *colorobj;

    /* Sanity check */
    if ( (self->enginename != CairoCFerBindName) &&
         (self->enginename != PyQtCairoCFerBindName) ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createColor: unexpected error, "
                            "self is not a valid CFerBind struct");
        return 0;
    }

    /* Verify valid fractions */
    if ( (0.0 > opaquefrac) || (opaquefrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_createColor: "
                             "invalid opaque fraction (%#.3f)", opaquefrac);
        return NULL;
    }
    if ( (0.0 > redfrac) || (redfrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_createColor: "
                             "invalid red fraction (%#.3f)", redfrac);
        return NULL;
    }
    if ( (0.0 > greenfrac) || (greenfrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_createColor: "
                             "invalid green fraction (%#.3f)", greenfrac);
        return NULL;
    }
    if ( (0.0 > bluefrac) || (bluefrac > 1.0) ){
        sprintf(grdelerrmsg, "cairoCFerBind_createColor: "
                             "invalid blue fraction (%#.3f)", bluefrac);
        return NULL;
    }

    colorobj = (CCFBColor *) FerMem_Malloc(sizeof(CCFBColor), __FILE__, __LINE__);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "cairoCFerBind_createColor: "
                            "out of memory for a CCFBColor structure");
        return NULL;
    }

    colorobj->id = CCFBColorId;
    colorobj->redfrac = redfrac;
    colorobj->greenfrac = greenfrac;
    colorobj->bluefrac = bluefrac;
    colorobj->opaquefrac = opaquefrac;

    return colorobj;
}

