	SUBROUTINE CD_SET_MODE(cdfid, mode, status)

* Set the netCDF mode to DATA or DEFINE
* but first check to see if it is already in the desired mode

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 10/92

* argument definitions:
*       cdfid   - returned pointer to open CDF file
*       mode    - parameter value from cdf_tmap.parm
*	status	- return status

* argument declarations:
	INTEGER	 cdfid, mode, status

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
        include 'tmap_format/cdf_tmap.parm'
        include 'tmap_format/tmap_errors.parm'
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
        INCLUDE 'TMAP_FORMAT:CDF_TMAP.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* local variable declarations:
      INTEGER cdf_mode, cdfstat

* must be remembered between calls
      SAVE cdf_mode

      IF ( mode .EQ. cdf_mode ) THEN
         RETURN      ! no change requested
      ELSEIF ( mode .EQ. pcd_mode_define ) THEN
* ... enter define mode
         CALL NCREDF( cdfid, cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5100
      ELSEIF ( mode .EQ. pcd_mode_data ) THEN
* ... enter data mode
         CALL NCENDF( cdfid, cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5100
      ENDIF

* save the specified mode (the only action taken when mode < 0)
      cdf_mode = ABS( mode ) 

* successful completion
      status = merr_ok
      RETURN

* error messages
 5100 CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_SET_MODE', cdfid,
     .     no_varid, 'could not change CDF mode', ' ', *5000 )

* error exit
 5000 RETURN
	END
