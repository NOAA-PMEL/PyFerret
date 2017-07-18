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



/*
 * save_arg_pointers.c
 *
 * Cache pointers passed from FORTRAN.
 * Later these pointers can be used to access the arrays in SUBROUTINES
 * which did not get these args passed.
 *
 * programmer - steve hankin
 * NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
 *
 * revision 0.0 - 3/5/97
 * 3/05 *acm* - new routines for SHADE/MOD curvilinear data: 
 *              curv_coord_add and curv_coord_range
 * 1/12 *acm* - Ferret 6.8 ifdef double_p for double-precision ferret, see the
 *              definition of macro DFTYPE in ferret.h
 */

#include "ferret.h"
 
/* global cache pointer */
DFTYPE *xpos_cache, *ypos_cache;

void FORTRAN(save_arg_pointers)( DFTYPE *xpos_arr, DFTYPE *ypos_arr )
{
  extern DFTYPE *xpos_cache, *ypos_cache;
  xpos_cache = xpos_arr;
  ypos_cache = ypos_arr;

  return;
}

/*******************/

/* prototype for FORTRAN subroutine to be called */
void FORTRAN(curv_coord_sub) ( DFTYPE *xi, DFTYPE *yi, int *n,
                      DFTYPE *xpos_cache, DFTYPE *ypos_cache,
                      DFTYPE *xinv, DFTYPE *yinv, int *status );

/* prototype for FORTRAN subroutine to be called */
void FORTRAN(curv_coord_add_sub) ( DFTYPE *xi, DFTYPE *yi, int *n,
                      DFTYPE *xpos_cache, DFTYPE *ypos_cache,
                      DFTYPE *xinv, DFTYPE *yinv, DFTYPE *xadd,
		              int *first, int*xfield_is_modulo, int *status );

/* prototype for FORTRAN subroutine to be called */
void FORTRAN(curv_coord_range_sub) ( DFTYPE *uc, DFTYPE *xpos_cache, DFTYPE *ypos_cache, 
                      int *ilo, int *ihi, int *jlo, int *jhi, 
		      int *status );

/*******************/

/* curv_coord: this routine, called from FORTRAN, will make the
arrays of X and Y positions available
 */

void FORTRAN(curv_coord)(DFTYPE *xi, DFTYPE *yi, int *n,
		DFTYPE *xinv, DFTYPE *yinv, int *status)
{
  extern DFTYPE *xpos_cache, *ypos_cache;
  FORTRAN(curv_coord_sub)( xi, yi, n, xpos_cache, ypos_cache, xinv, yinv, status );
  return;
}

/*******************/

/* curv_coord_add: this routine, called from FORTRAN, will make the
arrays of X and Y positions available, with an offset in the X coords
*/

void FORTRAN(curv_coord_add)(DFTYPE *xi, DFTYPE *yi, int *n,
		 DFTYPE *xinv, DFTYPE *yinv, DFTYPE *xadd, int *first, int*xfield_is_modulo, int *status)
{
  extern DFTYPE *xpos_cache, *ypos_cache;
  FORTRAN(curv_coord_add_sub)(  xi, yi, n, xpos_cache, ypos_cache, xinv, yinv, xadd, first, xfield_is_modulo, status );

  return;
}

/*******************/

/* curv_coord_range: this routine, called from FORTRAN, will compute
   range within Y coordinates actually needed for the vlimits 
   that were requested (x may be modulo so we need full range).
 */

void FORTRAN(curv_coord_range)(DFTYPE *uc, int *ilo, int *ihi, int *jlo, int *jhi, int *status)
{
  extern DFTYPE *xpos_cache, *ypos_cache;
  FORTRAN(curv_coord_range_sub)(uc, xpos_cache, ypos_cache, ilo, ihi, jlo, jhi, status );

  return;
}
