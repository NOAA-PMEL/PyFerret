	SUBROUTINE CD_RD_R8_ARR( cdfid, varid, start, n, dtype, errstr,
     .                           dvals, status )

* read a series of values from a netCDF variable and return them as 
* double prec.
* The variable must be a simple array (e.g. an axis)

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

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
      INTEGER cdfid, varid, start, n, dtype, status
      REAL*8  dvals(n)
      CHARACTER*(*) errstr

#ifdef unix
      include 'tmap_format/netcdf.decl'
      include 'tmap_format/netcdf.inc'
      include 'tmap_format/tmap_errors.parm'
#else
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
      INCLUDE 'TMAP_FORMAT:NETCDF.INC'
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* internal variable declarations:
      INTEGER cdfstat

* read the value
      CALL NCVGT( cdfid, varid, start, n, dvals, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_R8_ARR', cdfid, varid,      
     .       errstr, no_errstring, *5000 )

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
