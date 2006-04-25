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

    v540: 11/01 *kob* - change passed in values for start and count
                        to temp variables which are declared
			compatibly with the passed in types from the 
			fortran routine CD_READ.F.  Once in, need to 
			convert these to the proper type that the 
			netcdf c routines expect.  This may vary from
			o.s. to o.s. depending on what, for example,
			size_t and ptrdiff_t are typdef'd as.  They are
			typedef'd as different things under solaris and 
			compaq Tru64, for example



    compile this with
    cc -c -g -I/opt/local/netcdf-3.4/include cd_write_var_sub.c
*/ 

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <stdlib.h>
#include <wchar.h>
#include <stdio.h>
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
				int *dims, int *tmp_start, int *tmp_count, 
				int *strdim, void *dat, int *cdfstat )

{
  /* convert FORTRAN-index-ordered, FORTRAN-1-referenced ids, count,
     and start to C equivalent

     *kob*  11/01 need start and count  variables of the same type
                  as is predfined for each O.S.

     V542: 11/02 *acm*  Need start and count to be length [5] to allow for
			string dimension.

     V600:  2/06 *acm* Write more data types to netcdf files : Note that compiler
                       warnings may be seen about data type inconsistencies in the
					   calls to nc_put_vara_float. This is ok; see the comments at
					   the end of this file
  */

  size_t start[5], count[5];
  int tmp, i, maxstrlen;
  size_t bufsiz;
  char *pbuff;
  int ndim = *dims - 1; /* C referenced to zero */  
/*  int ndim = *dims ; /* C referenced to zero */
  int vid = *varid;
  int did = *strdim;
  vid--;
  did--;

  /* cast passed in int values (from fortran) to proper types, which can
     be different depending on o.s       *kob* 11/01 */
  for (i=0;i<=ndim+1;i++) {
    start[i] = (size_t)tmp_start[i];
    count[i] = (size_t)tmp_count[i];
  }

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


  switch (*vartyp) {
  case NC_CHAR:

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
  break;


  /* DOUBLE data */
  case NC_DOUBLE:
    *cdfstat = nc_put_vara_float(*cdfid, vid,
				 start, count, (double*) dat);
  break;

  /* FLOAT data */
  case NC_FLOAT:
    *cdfstat = nc_put_vara_float(*cdfid, vid,
				 start, count, (float*) dat);
  break;

  /* INT data */
  case NC_INT:
    *cdfstat = nc_put_vara_float(*cdfid, vid,
				 start, count, (int*) dat);
  break;

  /* SHORT data */
  case NC_SHORT:
    *cdfstat = nc_put_vara_float(*cdfid, vid,
				 start, count, (short*) dat);
  break;

  /* BYTE data */
  case NC_BYTE:
    *cdfstat = nc_put_vara_float(*cdfid, vid,
				 start, count, (char*) dat);
  break;

  default:
  break;
  }

  return;
}

/*
From a netcdf man page found on-line at

http://129.89.70.230/cgi-bin/IMT/wwwman?topic=netcdf(3)&msection=1

  int nc_put_vara_text(int ncid,  int  varid,  const  size_t  start[],	const
       size_t count[], const char out[])

  int nc_put_vara_uchar(int ncid, int  varid,  const  size_t  start[],	const
       size_t count[], const unsigned char out[])

  int nc_put_vara_schar(int ncid, int  varid,  const  size_t  start[],	const
       size_t count[], const signed char out[])

  int nc_put_vara_short(int ncid, int  varid,  const  size_t  start[],	const
       size_t count[], const short out[])

  int nc_put_vara_int(int ncid, int varid, const size_t start[], const size_t
       count[], const int out[])

  int nc_put_vara_long(int ncid,  int  varid,  const  size_t  start[],	const
       size_t count[], const long out[])

  int nc_put_vara_float(int ncid, int  varid,  const  size_t  start[],	const
       size_t count[], const float out[])

  int nc_put_vara_double(int ncid, int varid,  const  size_t  start[],	const
       size_t count[], const double out[])

       Writes an array section of values into a netCDF variable	 of  an	 open
       netCDF  dataset,	 which	must  be  in data mode.	 The array section is
       specified by the start and count vectors, which give the starting  in-
       dex  and	 count	of values along each dimension of the specified vari-
       able.  The type of the data is specified in the function name  and  is
       converted to the external type of the specified variable, if possible,
       otherwise an NC_ERANGE error is returned.

*/
