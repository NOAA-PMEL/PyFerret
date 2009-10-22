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
     Ansley Manke 9/2/2009
	cd_read_dp.c -- this routine hides the call to netCDF routine
    nc_get_vara_double. In netcdf4.1beta2, the call in CD_RD_R8_ARR to the 
	fortran NCVGT fails.
*/ 

#include <wchar.h>
/*#include <stdio.h>*/
#include <stdlib.h>
#include <netcdf.h>
#include <assert.h>
#include <stdio.h> 
#include <errno.h>


#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

void FORTRAN(cd_read_dp) (int *cdfid, int *varid, int *dims, 
			   int *tmp_start, int *tmp_count, 
			   void *dat, int *cdfstat )
{

  /* convert FORTRAN-index-ordered, FORTRAN-1-referenced ids, count,
     and start to C equivalent

     *kob* need start,count,stride and imap variables of the same type
           as is predfined for each O.S.
  */
  size_t start[5], count[5];

  int i, ndimsp, *dimids;
  size_t bufsiz, tmp;
  int ndim = 0;
  int indim = *dims;
  int vid = *varid;
  int ncid = *cdfid;
  nc_type vtyp;

	if (*dims > 0)
		ndim = *dims - 1; /* C referenced to zero */

  /* cast passed in int values (from fortran) to proper types, which can
     be different depending on o.s       *kob* 11/01 */
  for (i=0; i<5; i++) {
    start[i] = (size_t)tmp_start[i];
    count[i] = (size_t)tmp_count[i];
  }

  /* change FORTRAN indexing and offsets to C */
  vid--;
  for (i=0; i<=ndim; i++)
		{
			if (start[i] > 0)
				start[i]--;
		}

	if (ndim > 0)
		{
			for (i=0; i<=ndim/2; i++) 
				{
					tmp = count[i];
					count[i] = count[ndim-i];
					count[ndim-i] = tmp;
					tmp = start[i];
					start[i] = start[ndim-i];
					start[ndim-i] = tmp;
				}
		}
  /* get the type of the variable on disk */
  *cdfstat = nc_inq_vartype(*cdfid, vid, &vtyp);
  if (*cdfstat != NC_NOERR) return;

  /* read the data */
  *cdfstat = nc_get_vara_double (ncid, vid, start, count, (double*) dat); 

  return;
}

