	SUBROUTINE CD_RD_R8_1(cdfid,varid,index,dtype,errstr,dval,status)

* read a single value from a netCDF variable and return it as a double prec.
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
