#line 1  "finqpixel.fc"
/*
 *		Copyright IBM Corporation 1989
 *
 *                      All Rights Reserved
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appear in all copies and that
 * both that copyright notice and this permission notice appear in
 * supporting documentation, and that the name of IBM not be
 * used in advertising or publicity pertaining to distribution of the
 * software without specific, written prior permission.
 *
 * IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
 * ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
 * IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
 * ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
 * ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *
 *
 * FORTRAN to C binding for XGKS
 *
 * GKS inquire pixel functions :
 *	gqpxad_
 *	gqpxa_
 *	gqpx_
 *
 * David Berkowitz
 * Bruce Haimowitz
 * Todd Gill
 * TCS Development
 * Cambridge MA
 *
 * August 31 1988
 *
 * August 27, 2004 ACM Put a test in gqpxad for whether the workstation type
 *                     is X_WIN. If not, then do not call ginqpixelarraydim.
 *                     This lets us call the routine when in -gif  or -batch
 *                     mode, without displaying an error message.
 */

/*LINTLIBRARY*/

#include "udposix.h"
#include <stdlib.h>
#include "xgks.h"
#include "fortxgks.h"


/* extra definitions from inqpixel.c for skipping  
 * the call to ginqpixelarraydim when in gif mode*/

#include <wchar.h>
#include <wchar.h>
#include "gks_implem.h"

/* end of extra definitions from inqpixel.c: */

#ifndef lint
    static char afsid[] = "$__Header$";
    static char rcsid[] = "$Id$";
#endif


/*
 * gqpxad - Inquire Pixel Array Dimensions
 *
 * int	*wkid		- pointer to workstation identifier
 * float *px,*py	- pointer to upper left corner in world coordinates
 * float *qx,*qy	- pointer to lower right corner in world coordinates
 * int	*errind		- pointer to error indicator
 * int	*n,m		- pointer to dimensions of pixel array
 *
 * Returns: ANSI standard errors for this function.
 *
 * See also: ANSI standard p.190
 */
#line 81  "finqpixel.fc"
gqpxad_(wkid, px, py, qx, qy, errind, n, m)
    int       *wkid;
    float     *px;
    float     *py;
    float     *qx;
    float     *qy;
    int       *errind;
    int       *n;
    int       *m; 
{
#line 91  "finqpixel.fc"
    Grect           rect;
    Gipoint         dim;
    WS_STATE_PTR    ws;

    debug(("Inquire Pixel Array Dimensions \n"));

/*  Bug fixed in assigning ll and ur coordinate of rectangle *jd* 12.23.93 */
/*  rect.ll.y = (Gfloat) *py; This is ul */
/*  rect.ur.y = (Gfloat) *qy; This is lr */

    rect.ll.x = (Gfloat) *px;
    rect.ll.y = (Gfloat) *qy;
    rect.ur.x = (Gfloat) *qx;
    rect.ur.y = (Gfloat) *py;

	
/* extra calls  from inqpixel.c for skipping  
 * the call to ginqpixelarraydim when in gif mode*/

     /* check for proper operating state */
    GKSERROR((xgks_state.gks_state == GGKCL || xgks_state.gks_state == GGKOP), 
	     7, errginqpixelarraydim);

    /* check for invalid workstation id */
    GKSERROR((!VALID_WSID(*wkid)), 20, errginqpixelarraydim);

    /* check if this workstation is opened */
    GKSERROR(((ws = OPEN_WSID(*wkid)) == NULL), 25, errginqpixelarraydim);

    if (ws->ewstype != X_WIN)
    return;

/* end of extra code from inqpixel.c: */

    if (*errind = ginqpixelarraydim((Gint) *wkid, &rect, &dim))
	return;

    *n = (int) dim.x;
    *m = (int) dim.y;
}


/*
 * gqpxa - Inquire Pixel Array
 *
 * int	*wkid		- pointer to workstation identifier
 * float *px,*py	- pointer to upper left corner in world coordinates
 * int	*dimx,*dimy	- pointers to dimensions of colour index array
 * int	*isc,*isr	- pointers to start column, start row
 * int	*dx,*dy		- pointers to size of requested pixel array
 * int	*errind		- pointer to error indicator
 * int	*invval		- pointer to presence of invalid values (GABSNT, GPRSNT)
 * int	*colia[dimx,dimy] - pointer to colour index array
 *
 * Returns: Error 2002 in addition to ANSI standard errors for this function.
 *
 * See also: ANSI standard p.191
 */
#line 150  "finqpixel.fc"
gqpxa_(wkid, px, py, dimx, dimy, isc, isr, dx, dy, errind, invval, colia)
    int       *wkid;
    float     *px;
    float     *py;
    int       *dimx;
    int       *dimy;
    int       *isc;
    int       *isr;
    int       *dx;
    int       *dy;
    int       *errind;
    int       *invval;
    int       *colia; 
{
#line 164  "finqpixel.fc"
    Gpoint          point;
    Gipoint         dimen;
    Gpxarray        pxarr;

    debug(("Inquire Pixel Array \n"));

    dimen.x = *dx;
    dimen.y = *dy;

    if (*dx > *dimx || *dy > *dimy) {
	*errind = 2001;
	(void) gerrorhand(2001, errginqpixelarray, (errfp));
	return;
    }
    point.x = *px;
    point.y = *py;

    if (*errind = ginqpixelarray((Gint) *wkid, &point, &dimen, &pxarr))
	return;


    *invval = (int) pxarr.covalid;

    /*
     * This array does not need to be transposed; however, we must
     * fit the dx-by-dy array obtained into the dimx-by-dimy array provided.
     */
    {
	int             col, row;

	for (row = 0; row < *dy; row++)
	    for (col = 0; col < *dx; col++)
		*(colia + (row * *dimx + col)) =
		    *(pxarr.array + (row * *dx + col));
    }

    free((voidp) pxarr.array);
}


/*
 * gqpx - Inquire Pixel
 *
 * int	*wkid		- pointer to workstation identifier
 * float *px,*py	- pointer to upper left corner in world coordinates
 * int	*errind		- pointer to error indicator
 * int	*coli		- pointer to colour index
 *
 * Returns: ANSI standard errors for this function.
 *
 * See also: ANSI standard p.191
 */
#line 217  "finqpixel.fc"
gqpx_(wkid, px, py, errind, coli)
    int       *wkid;
    float     *px;
    float     *py;
    int       *errind;
    int       *coli; 
{
#line 224  "finqpixel.fc"
    Gpoint          ppoint;

    ppoint.x = *px;
    ppoint.y = *py;

    *errind = ginqpixel((Gint) *wkid, &ppoint, (Gint*) coli);
}
