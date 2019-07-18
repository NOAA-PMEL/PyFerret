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
    cd_rd_str_1_sub.c from cd_read_sub, just read one string from a 1-D var
	The calling routine has checked the grid to make sure it's 1-D.

#include <stddef.h>  /* size_t, ptrdiff_t; gfortran on linux rh5*/
#include <wchar.h>
#include <stdlib.h>
#include <netcdf.h>
#include "fmtprotos.h"
#include "list.h"
#include "NCF_Util.h"
#include "FerMem.h"

void FORTRAN(cd_rd_str_1_sub)(int *cdfid, int *varid, int *tmp_start,
                          char* buff, int *slen, int *cdfstat)
{


  size_t start[2], count[2];

  int i, ndimsp, *dimids;
  size_t bufsiz, tmp, maxstrlen;
  char *pbuff;
  int ndim = 0;
  int vid = *varid;
  nc_type vtyp;

  start[0] = *tmp_start;
  count[0] = 1;

  start[0]--;

  /* get the type of the variable on disk */
  vid--;
  *cdfstat = nc_inq_vartype(*cdfid, vid, &vtyp);
  if (*cdfstat != NC_NOERR) {
      return;
  }
  /* get the data */
  if (vtyp == NC_CHAR) {

      *cdfstat = nc_inq_varndims (*cdfid, vid, &ndimsp);
      if (*cdfstat != NC_NOERR) {
          return;
      }
      dimids = (int *) FerMem_Malloc(sizeof(int) * ndimsp, __FILE__, __LINE__);
      if ( dimids == NULL )
          abort();
      ndimsp--;
      *cdfstat = nc_inq_vardimid (*cdfid, vid, dimids);
      if (*cdfstat != NC_NOERR) {
          return;
      }
      *cdfstat = nc_inq_dimlen (*cdfid, dimids[ndimsp], &bufsiz);
      if (*cdfstat != NC_NOERR) {
          return;
      }
      FerMem_Free(dimids, __FILE__, __LINE__);
      maxstrlen = bufsiz;
      pbuff = (char *) FerMem_Malloc(sizeof(char) * bufsiz, __FILE__, __LINE__);
      if ( pbuff == NULL )
         abort();
      /* update variable dimensions to include string dimension */
      start[ndimsp]  = 0;
      count[ndimsp]  = maxstrlen;

      *cdfstat = nc_get_vara_text (*cdfid, vid, start, count, pbuff);
      strcpy ( buff, pbuff );
	  *slen= strlen(buff);
	  if (*slen > bufsiz) *slen = bufsiz;
	  }

  /* Numeric data. return some error */
   else
	{
	   *cdfstat = -9;
	  }

  return;
}

