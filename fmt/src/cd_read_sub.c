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
    cd_read_sub.c -- this routine hides the call to netCDF routine
    NF_GET_VARM_<type> allowing last minute modifications to the
    call arguments for Ferret interactive data analysis program

    programmer - steve hankin
    NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

    revision history:
    V533: 6/01 *sh* - original

    v540: 11/01 *kob* - change passed in values for start,count,stride
                        and imap to temp variables which are declared
			compatibly with the passed in types from the 
			fortran routine CD_READ.F.  Once in, need to 
			convert these to the proper type that the 
			netcdf c routines expect.  This may vary from
			o.s. to o.s. depending on what, for example,
			size_t and ptrdiff_t are typdef'd as.  They are
			typedef'd as different things under solaris and 
			compaq Tru64, for example


     V542: 11/02 *acm*  Need start and count to be length [5] to allow for
			string dimension.  Same for stride[5], imap[5]

    compile this with
    cc -c -g -I/opt/local/netcdf-3.4/include cd_read_sub.c
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
void tm_unblockify_ferret_strings(void *dat, char *pbuff,
				int bufsiz, int outstrlen);

void FORTRAN(cd_read_sub) (int *cdfid, int *varid, int *dims, 
			   int *tmp_start, int *tmp_count, 
			   int *tmp_stride, int *tmp_imap,
			   void *dat, int *cdfstat )
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
  char *pbuff;
  int ndim = *dims - 1; /* C referenced to zero */
  int vid = *varid;
  nc_type vtyp;


  /* cast passed in int values (from fortran) to proper types, which can
     be different depending on o.s       *kob* 11/01 */
  for (i=0; i<=ndim; i++) {
    start[i] = (size_t)tmp_start[i];
    count[i] = (size_t)tmp_count[i];
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

  /* write out the data */
  if (vtyp == NC_CHAR) {
    /* Read into a buffer area with the multi-dimensiona array of strings
       packed into a block. Unpack it into "dat".
       The "dat" variables is a pointer to an array of string pointers
       where the string pointers are spaced 8 bytes apart
    */
    
      *cdfstat = nc_inq_varndims (*cdfid, vid, &ndimsp);
      if (*cdfstat != NC_NOERR) return;
      dimids =  (int *) malloc(sizeof(int) * ndimsp);
      assert(dimids);
      ndimsp--;
      *cdfstat = nc_inq_vardimid (*cdfid, vid, dimids);
      if (*cdfstat != NC_NOERR) return;
      *cdfstat = nc_inq_dimlen (*cdfid, dimids[ndimsp], &bufsiz);
      if (*cdfstat != NC_NOERR) return;
      free(dimids);
      maxstrlen = bufsiz;
      for (i=0; i<=ndim; i++) bufsiz *= count[i];
      pbuff = (char *) malloc(sizeof(char) * bufsiz);
      assert(pbuff);

      /* update variable dimensions to include string dimension */
      start[ndimsp]  = 0;
      count[ndimsp]  = maxstrlen;
      stride[ndimsp] = 1;
      for (i=0; i<=ndim; i++) imap[i] *= maxstrlen;      
      imap[ndimsp] = 1;

      *cdfstat = nc_get_varm_text (*cdfid, vid, start,
				    count, stride, imap, pbuff);
      tm_unblockify_ferret_strings(dat, pbuff, bufsiz, maxstrlen);
      free(pbuff);


  /* FLOAT data */
  } else
    *cdfstat = nc_get_varm_float (*cdfid, vid, start,
				  count, stride, imap, (float*) dat);

  return;
}

