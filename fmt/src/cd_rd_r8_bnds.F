	SUBROUTINE CD_RD_R8_BNDS( cdfid, varid, start, npts, dtype, errstr,
     .                           dvals, d1s, d1e, d2s, d2e, status )

*
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
*
* read a series of values from a netCDF variable and return them as 
* double prec.
* The variable must be a simple array (e.g. an axis)

* From cd_rd_r8_arr.F
* 6/2004
* 6/04  *acm* for compiler on rh7, need to pass dimensions of array dvals 
*        into this routine as scalar integers not array elements.

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       varid    - variable id in netCDF file
*       start    - start subscript of netCDF variable to return
*       n        - number of values to return
*	dtype    - data type of variable
*       errstr   - identifying string to be printed with error messages
*       dvals    - double precision values returned
*       status   - TMAP library return status

* argument declarations
      INTEGER cdfid, varid, start(2), npts(2), dtype, status,
     .        d1s, d1e, d2s, d2e
c      REAL*8  dvals( start(1):npts(1), start(2):npts(2) )
      REAL*8  dvals( d1s:d1e, d2s:d2e )
      CHARACTER*(*) errstr

#ifdef unix
      include 'netcdf.decl'
#include "netcdf.inc"
      include 'tmap_errors.parm'
#else
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
#include "netcdf.inc"
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* internal variable declarations:
      INTEGER cdfstat, n

* read the value
      CALL NCVGT( cdfid, varid, start, npts, dvals, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_R8_ARR', cdfid, varid,      
     .       errstr, no_errstring, *5000 )

      n = (npts(1)-start(1)+1) * (npts(2)-start(2)+1)

* convert data types
      IF (dtype.EQ.NCDOUBLE) THEN
         RETURN
      ELSEIF (dtype.EQ.NCBYTE .OR. dtype.EQ.NCCHAR) THEN
         CALL TM_I1_to_R8( dvals, dvals, n)
      ELSEIF (dtype.EQ.NCSHORT) THEN
         CALL TM_I2_TO_R8( dvals, dvals, n)
      ELSEIF (dtype.EQ.NCLONG ) THEN
         CALL TM_I4_TO_R8( dvals, dvals, n)
      ELSE
         CALL TM_R4_TO_R8( dvals, dvals, n)
      ENDIF

      status = merr_ok

 5000 RETURN
      END
