/* Read a DOUBLE hyperslab from a netCDF file into a FLOAT array.
   Since the array to be read is bigger than the amount of memory
   we can count on being available in the calling program buffer
   we will allocate temporary storage and use it to buffer the read.
*/

/* compilation may require "-I/usr/local/include" (to find netcdf.h) */

/* Call from FORTRAN using

     CALL CD_RD_R8_AS_R4( cdfid, varid, start, count, ndim, array, cdfstat )
*/

/*#include "tmap_format/netcdf.h"*/
#include <netcdf.h>      /* usually in /usr/local/include */
/* Suns need to include stdio.h to get definition for NULL */
/* #include <stdlib.h> */
#include <stdio.h>

#ifdef NO_ENTRY_NAME_UNDERSCORES
void cd_rd_r8_as_r4(cdfid,
#else
void cd_rd_r8_as_r4_(cdfid,
#endif
		    varid,
		    start,
		    count,
		    ndim,
		    values,
		    cdfstat)

int *cdfid;
int *varid;
int start[4];
int count[4];
int *ndim;
float *values;
int *cdfstat;/*returns one of: ncnoerr, netCDF status, or -1 (malloc failed)*/ 

{

  int  idim, i, npts, rcode;
  long cstart[4], ccount[4];
  double *dvals;

/* change the start/count values to C ordering and the start to zero offset */
  for (idim=0; idim<*ndim; idim++){
    cstart[idim] = (long) (start[(*ndim)-1-idim] - 1);
    ccount[idim] = (long) (count[(*ndim)-1-idim]);
  }
/* the total number of data points */
  for (npts=1,idim=0; idim<*ndim; idim++) npts *= count[idim];

/* allocate memory for the double precision hyperslab */
  dvals = (double *) malloc(8*npts);
  if ( dvals == NULL ) {
    *cdfstat = -1;
    return;
  }

/* read the data */
  rcode = ncvarget(*cdfid,*varid-1,cstart,ccount,dvals);
  if ( rcode == -1 ) {
    free(dvals);
    *cdfstat = ncerr;     /* global var from netcdf.h */
    return;
  }
  
/* convert to single precision */
  for (i=0; i<npts; i++) values[i] = (float) dvals[i];

/* successful completion */
  free(dvals);
  *cdfstat = NC_NOERR;
  return;
}
