	SUBROUTINE CD_RD_R8_1(cdfid,varid,index,dtype,errstr,dval,status)

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
* The variable must be a simple array (e.g. an axis)

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       varid    - variable id in netCDF file
*       index    - subscript of array to return
*	dtype    - data type of variable
*       errstr   - identifying string to be printed with error messages
*       vname    - name of variable owning attribute (used in error msgs)
*       dval     - double precision value returned
*       status   - TMAP library return status

* argument declarations
      INTEGER cdfid, varid, index, dtype, status
      REAL*8  dval
      CHARACTER*(*) errstr

#ifdef unix
      include 'tmap_format/netcdf.decl'
      include 'tmap_format/netcdf.inc'
      include 'tmap_format/tmap_errors.parm'
#  ifdef sun
      BYTE i1buff
#  else
      INTEGER*1 i1buff
#  endif
#else
      BYTE i1buff
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
      INCLUDE 'TMAP_FORMAT:NETCDF.INC'
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* internal variable declarations:
      INTEGER cdfstat

* equivalent buffer arrays for various data types
      INTEGER*2 i2buff
      INTEGER*4 i4buff
      REAL      r4buff
      REAL*8    r8buff
      COMMON /xrd_r8_1/ r8buff     ! debugger seems to need this
      EQUIVALENCE (r8buff,i1buff),
     .            (r8buff,i2buff),
     .            (r8buff,i4buff),
     .            (r8buff,r4buff)

* read the value
      CALL NCVGT1( cdfid, varid, index, r8buff, cdfstat )

      IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_R8_1', cdfid, varid,
     .       errstr, no_errstring, *5000 )

* convert data types
      IF (dtype.EQ.NCBYTE .OR. dtype.EQ.NCCHAR) THEN
         dval = i1buff
      ELSEIF (dtype.EQ.NCSHORT) THEN
         dval = i2buff
      ELSEIF (dtype.EQ.NCLONG ) THEN
         dval = i4buff
      ELSEIF (dtype.EQ.NCFLOAT ) THEN
         dval = r4buff
      ELSE
         dval = r8buff
      ENDIF

      status = merr_ok

 5000 RETURN

      END
