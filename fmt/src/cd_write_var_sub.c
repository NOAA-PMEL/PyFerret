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
    cd_write_var_sub.c -- this routine hides the call to netCDF routine
    NCVPT allowing last minute modifications to the call arguments


    for Ferret interactive data analysis program

    programmer - steve hankin
    NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

    revision history:
    V533: 6/01 *sh* - original

    compile this with
    cc -c -g -I/opt/local/netcdf-3.4/include cd_write_var_sub.c
*/ 

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
void tm_blockify_ferret_strings(void *dat, char *pbuff,
				int bufsiz, int outstrlen);

void FORTRAN(cd_write_var_sub) (int *cdfid, int *varid, int *vartyp,
				int *dims, size_t start[], size_t count[], 
				int *strdim, void *dat, int *cdfstat )

{
  /* convert FORTRAN-index-ordered, FORTRAN-1-referenced ids, count,
     and start to C equivalent
  */
  int tmp, i, maxstrlen;
  size_t bufsiz;
  char *pbuff;
  int ndim = *dims - 1; /* C referenced to zero */
  int vid = *varid;
  int did = *strdim;
  vid--;
  did--;
  for (i=0; i<=ndim; i++)
    start[i]--;
  for (i=0; i<=ndim/2; i++) {
    tmp = count[i];
    count[i] = count[ndim-i];
    count[ndim-i] = tmp;
    tmp = start[i];
    start[i] = start[ndim-i];
    start[ndim-i] = tmp;
  }

  /* write out the data */
  if (*vartyp == NC_CHAR) {
    /* Create a buffer area with the multi-dimensiona array of strings
       packed into a block.
       The "dat" variables is a pointer to an array of string pointers
       where the string pointers are spaced 8 bytes apart
    */
      *cdfstat = nc_inq_dimlen (*cdfid, did, &bufsiz);
      if (*cdfstat != NC_NOERR) return;
      maxstrlen = bufsiz;
      for (i=0; i<=ndim; i++) bufsiz *= count[i];
      pbuff = (char *) malloc(sizeof(char) * bufsiz);
      assert(pbuff);
      tm_blockify_ferret_strings(dat, pbuff, bufsiz, maxstrlen);

      /* update variable dimensions to include string dimension */
      start[*dims] = 0;
      count[*dims] = maxstrlen;

      *cdfstat = nc_put_vara_text(*cdfid, vid,
				 start, count, pbuff);
      free(pbuff);


  /* FLOAT data */
  } else
    *cdfstat = nc_put_vara_float(*cdfid, vid,
				 start, count, (float*) dat);

  return;
}

