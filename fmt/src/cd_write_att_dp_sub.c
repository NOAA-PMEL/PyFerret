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
    cd_write_att_dp_sub.c
	(from cd_write_var_sub.c) -- write attribute which comes into the routine
	                             as type double

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

void FORTRAN(cd_write_att_dp_sub) (int *cdfid, int *varid, char* attname, int *attype,
				                   int *nval, void *val, int *status )

{
  /*
     V600:  2/06 *acm* Write correct atttr type to netcdf files : Note that compiler
                       warnings may be seen about data type inconsistencies in the
					   calls to nc_put_att_double. This is ok; netcdf library does conversion.
					   
  */
  
  int vid = *varid;
  vid--;

  /* write out the data */


  switch (*attype) {

  /* DOUBLE attr */
  case NC_DOUBLE:
      *status= nc_put_att_double (*cdfid, vid, attname, *attype,
              *nval, (double*) val);
  break;

  /* FLOAT attr */
  case NC_FLOAT:
      *status= nc_put_att_double (*cdfid, vid, attname, *attype,
              *nval, (float*) val);
  break;

  /* INT attr */
  case NC_INT:
	  *status= nc_put_att_double (*cdfid, vid, attname, *attype,
              *nval, (int*) val);
  break;

  /* SHORT attr */
  case NC_SHORT:
	  *status= nc_put_att_double (*cdfid, vid, attname, *attype,
              *nval, (short*) val);
  break;

  /* Byte attr */
  case NC_BYTE:
	  *status= nc_put_att_double (*cdfid, vid, attname, *attype,
              *nval, (char*) val);
  break;

  default:
  break;
  }

  return;
}
