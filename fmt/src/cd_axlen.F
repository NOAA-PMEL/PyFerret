	INTEGER FUNCTION CD_AXLEN ( cdfid, dimid, status )

* get the length of an axis (netCDF coordinate variable)
* the axis is presumed to exist and to be a 1D variable

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 4/94 *sh*,*kob* inserted return value for error exit case - avoid
*		  compiler error msg

* argument definitions:
*	cdfid	- netCDF id number for already opened CDF file
*       dimid   - dimension id of coordinate variable in netCDF file
*       status  - returned status code

* argument declarations
      INTEGER	    cdfid, dimid, status

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
* *kob* explicit definitions not allowed on decl line in linux...moved it to parameter line
      LOGICAL CD_GET_ATTVAL, got_it, do_warn
      PARAMETER (do_warn = .TRUE.)
      INTEGER NCVID, TM_LENSTR1, cdfstat, nval, npts, nlen, axid
      CHARACTER*24 axname
      REAL rlen

* initialize in case of error exit (4/94)
      CD_AXLEN = 0

* get the basic information about this dimension
      CALL NCDINQ( cdfid, dimid, axname, npts, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* get the basics about the axis by the same name
      nlen = TM_LENSTR1( axname )
      axid = NCVID( cdfid, axname(:nlen), cdfstat )

* check "true_size" in case this is a condensed axis representation
      got_it = CD_GET_ATTVAL( cdfid, axid, 'true_size',
     .                        do_warn, axname(:nlen), rlen, 1, nval)
      IF (got_it) npts = rlen

* successful completion
      CD_AXLEN = npts
      status = merr_ok
      RETURN

* error exit
 5000 RETURN
 5100 CALL TM_ERRMSG( cdfstat+pcdferr, status, 'CD_AXLEN', cdfid,
     .                no_varid, no_errstring, no_errstring, *5000 )
      END
