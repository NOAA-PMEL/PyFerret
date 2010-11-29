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
    cd_read_scale.c -- from cd_read_sub.c to read data as double
    precision, if that is how they are in the file, and apply the
    scale and offset values before converting to single precision.
    
  v6.01 9/06 *acm* use a malloc rather than a fixed buffer -> the data
              does not need to be 1-D
*/ 

#include <stddef.h>  /* size_t, ptrdiff_t; gfortran on linux rh5*/
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <assert.h>

#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

/* prototype */
void tm_scale_buffer(float *dat, double *dbuff,
			   float *offset, float *scale, float *bad,
			   int ntotal);

void FORTRAN(cd_read_scale) (int *cdfid, int *varid, int *dims, 
			   float *offset, float *scale, float* bad,
			   int *tmp_start, int *tmp_count, 
			   int *tmp_stride, int *tmp_imap,
			   void *dat, int *already_scaled, int *cdfstat, int *status)

{

  /* convert FORTRAN-index-ordered, FORTRAN-1-referenced ids, count,
     and start to C equivalent

     *kob* need start,count,stride and imap variables of the same type
           as is predfined for each O.S.
  */

  size_t start[5], count[5];
  ptrdiff_t stride[5], imap[5];

  int tmp, i, maxstrlen, ndimsp, *dimids;
  size_t bufsiz;
  int ndim = *dims - 1; /* C referenced to zero */
  int vid = *varid;
  nc_type vtyp;
  int n_sections;
  int ntotal;
  int scale_it;
  double *data_double;

  /* cast passed in int values (from fortran) to proper types, which can
     be different depending on o.s       *kob* 11/01 */
  ntotal = 1;
  for (i=0; i<=ndim; i++) {
    start[i] = (size_t)tmp_start[i];
    count[i] = (size_t)tmp_count[i];
	ntotal = ntotal * count[i];
    stride[i] = (ptrdiff_t)tmp_stride[i];
    imap[i] = (ptrdiff_t)tmp_imap[i];
  }


  /* change FORTRAN indexing and offsets to C */
  vid--;
  for (i=0; i<=ndim; i++)
    start[i]--;
  for (i=0; i<=ndim/2; i++) {
    tmp = count[i];
    count[i] = count[ndim-i];
    count[ndim-i] = tmp;
    tmp = start[i];
    start[i] = start[ndim-i];
    start[ndim-i] = tmp;
    tmp = stride[i];
    stride[i] = stride[ndim-i];
    stride[ndim-i] = tmp;
    tmp = imap[i];
    imap[i] = imap[ndim-i];
    imap[ndim-i] = tmp;
  }

  /* get the type of the variable on disk */
  *cdfstat = nc_inq_vartype(*cdfid, vid, &vtyp);
  if (*cdfstat != NC_NOERR) return;

  *status = 3;  /* merr_ok*/


  if (vtyp == NC_CHAR) 
	{
	  *status = 111;
	  return;
	}

  scale_it = 0;
  if (*offset != 0 || *scale != 1)
  { scale_it = 1;
  }
  if (vtyp == NC_DOUBLE && scale_it) 
  {

    /* Read into a buffer area as double precision,
	   and apply the scaling before converting to single precision 
	   in variable dat
    */

      data_double = (double *) malloc(ntotal * sizeof(double));
      assert(data_double);

	  *cdfstat = nc_get_varm_double (*cdfid, vid, start,
				  count, stride, imap, data_double);

      tm_scale_buffer ((float*) dat, data_double, offset,
         scale, bad, ntotal);
	  *already_scaled = 1;

	  free(data_double);
                  
  }
   
  /* read float data */
  else
  {
    *cdfstat = nc_get_varm_float (*cdfid, vid, start,
				  count, stride, imap, (float*) dat);
  }

  return;
}

/*  */
void tm_scale_buffer(float *dat, double *dbuff,
                     float *offset, float *scale, float *bad,
                     int ntotal)

{
        int j;
        double dbad;

        dbad = (double)*bad;
        for (j=0; j<ntotal; j++ )
        {
                if (dbuff[j] == dbad)
                        { dat[j] = *bad;
                        }
                else
                        {dat[j] = dbuff[j] * *scale + *offset;
                }
        }
    return;
}
