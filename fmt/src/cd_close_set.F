	SUBROUTINE CD_CLOSE_SET ( cdfid, status)

* Close a netCDF file 

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 6/92

* argument definitions:
*       cdfid   - returned pointer to open CDF file
*	status	- return status

* argument declarations:
	INTEGER	      cdfid, status

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
        include 'tmap_format/tmap_errors.parm'
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* local variable declarations:
      INTEGER cdfstat

* close the netCDF file
        CALL NCCLOS( cdfid, cdfstat )
        IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* successful completion
        status = merr_ok
        RETURN

* error exit
 5000   RETURN
 5100   CALL TM_ERRMSG(pcdferr+pcdferrmax,
     .         status, 'CD_CLOSE_SET', cdfid, -999,
     .         no_errstring, no_errstring, *5000 )
	END
