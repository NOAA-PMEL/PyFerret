/* save_arg_pointers.c

   Cache pointers passed from FORTRAN.
   Later these pointers can be used to access the arrays in SUBROUTINES
   which did not get these args passed.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/5/97
* 8/97 *kob* - had to add another ifdef check for entry_name_underscores
*              around call to curv_coord_sub
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
