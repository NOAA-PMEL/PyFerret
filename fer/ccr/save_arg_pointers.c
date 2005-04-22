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



/* save_arg_pointers.c

   Cache pointers passed from FORTRAN.
   Later these pointers can be used to access the arrays in SUBROUTINES
   which did not get these args passed.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/5/97
* 8/97 *kob* - had to add another ifdef check for entry_name_underscores
*              around call to curv_coord_sub
* 3/05 *acm* - new routines for SHADE/MOD curvilinear data: 
*              curv_coord_add and curv_coord_range
* compile with
*    cc -g -c save_arg_pointers.c
*  or
*    cc    -c save_arg_pointers.c
*/

 
/* global cache pointer */
float *xpos_cache, *ypos_cache;

#ifdef NO_ENTRY_NAME_UNDERSCORES
void save_arg_pointers( float *xpos_arr, float *ypos_arr )
#else
void save_arg_pointers_( float *xpos_arr, float *ypos_arr )
#endif

{
  extern float *xpos_cache, *ypos_cache;
  xpos_cache = xpos_arr;
  ypos_cache = ypos_arr;

  return;
}

/*******************/

/* prototype for FORTRAN subroutine to be called */
#ifdef NO_ENTRY_NAME_UNDERSCORES
void curv_coord_sub ( float *xi, float *yi, int *n,
                      float *xpos_cache, float *ypos_cache,
                      float *xinv, float *yinv,
		      int *status );
#else
void curv_coord_sub_( float *xi, float *yi, int *n,
                      float *xpos_cache, float *ypos_cache,
                      float *xinv, float *yinv,
		      int *status );
#endif


/* prototype for FORTRAN subroutine to be called */
#ifdef NO_ENTRY_NAME_UNDERSCORES
void curv_coord_add_sub ( float *xi, float *yi, int *n,
                      float *xpos_cache, float *ypos_cache,
                      float *xinv, float *yinv, float *xadd,
		      int *status );
#else
void curv_coord_add_sub_( float *xi, float *yi, int *n,
                      float *xpos_cache, float *ypos_cache,
                      float *xinv, float *yinv, float *xadd,
		      int *status );
#endif

/* prototype for FORTRAN subroutine to be called */
#ifdef NO_ENTRY_NAME_UNDERSCORES
void curv_coord_range_sub ( float *uc, float *xpos_cache, float *ypos_cache, 
                      int *ilo, int *ihi, int *jlo, int *jhi, 
		      int *status );
#else
void curv_coord_range_sub_( float *uc, float *xpos_cache,float *ypos_cache,  
                      int *ilo, int *ihi, int *jlo, int *jhi, 
		      int *status );
#endif

/*******************/

/* curv_coord: this routine, called from FORTRAN, will make the
arrays of X and Y positions available
 */

#ifdef NO_ENTRY_NAME_UNDERSCORES
void curv_coord(float *xi, float *yi, int *n,
		float *xinv, float *yinv, int *status)
#else
void curv_coord_(float *xi, float *yi, int *n,
		 float *xinv, float *yinv, int *status)
#endif
{
  extern float *xpos_cache, *ypos_cache;
#ifdef NO_ENTRY_NAME_UNDERSCORES
  curv_coord_sub( xi, yi, n, xpos_cache, ypos_cache, xinv, yinv, status );
#else
  curv_coord_sub_( xi, yi, n, xpos_cache, ypos_cache, xinv, yinv, status );
#endif
  return;
}

/*******************/

/* curv_coord_add: this routine, called from FORTRAN, will make the
arrays of X and Y positions available, with an offset in the X coords
*/

#ifdef NO_ENTRY_NAME_UNDERSCORES
void curv_coord_add(float *xi, float *yi, int *n,
		 float *xinv, float *yinv, float *xadd, int *status)
#else
void curv_coord_add_(float *xi, float *yi, int *n,
		 float *xinv, float *yinv, float *xadd, int *status)
#endif
{
  extern float *xpos_cache, *ypos_cache;
#ifdef NO_ENTRY_NAME_UNDERSCORES
  curv_coord_add_sub(  xi, yi, n, xpos_cache, ypos_cache, xinv, yinv, xadd, status );
#else
  curv_coord_add_sub_( xi, yi, n, xpos_cache, ypos_cache, xinv, yinv, xadd, status );
#endif
  return;
}

/*******************/

/* curv_coord_range: this routine, called from FORTRAN, will compute
   range within Y coordinates actually needed for the vlimits 
   that were requested (x may be modulo so we need full range).
 */

#ifdef NO_ENTRY_NAME_UNDERSCORES
void curv_coord_range(float *uc, int *ilo, int *ihi, int *jlo, int *jhi, int *status)
#else
void curv_coord_range_( float *uc, int *ilo, int *ihi, int *jlo, int *jhi, int *status)
#endif
{
  extern float *xpos_cache, *ypos_cache;
#ifdef NO_ENTRY_NAME_UNDERSCORES
  curv_coord_range_sub(uc, xpos_cache, ypos_cache, ilo, ihi, jlo, jhi, status );
#else
  curv_coord_range_sub_(uc, xpos_cache, ypos_cache, ilo, ihi, jlo, jhi, status );
#endif
  return;
}
