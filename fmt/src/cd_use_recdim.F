	LOGICAL FUNCTION CD_USE_RECDIM( cdfid, varid, status )

* Does the indicated variable use the netCDF record dim ?

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 6/92
* 15apr94 *kob* Need to initialzie cd_use_recdim in case of error 
*		exit
*
* 2jun94 *kob* initialization of cd_use_recdim needs to be boolean .FALSE
* 		instead of 0
* argument definitions:
*       cdfid   - returned pointer to open CDF file
*       varid   - netCDF variable id
*	status	- return status

* argument declarations:
	INTEGER	      cdfid, varid, status

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
      INTEGER ndims, nvars, ngatts, recdim, cdfstat, vartyp, nvdims,
     .        vdims(6), nvatts
      CHARACTER vname*32

*initialize
* 2jun94 *kob* changed initialization from 0 to .FALSE.
	CD_USE_RECDIM = .FALSE. 		!in case of error exit

* get vital statistics of file
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )
        IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* is there a record dimension
        IF ( recdim .EQ. -1 ) THEN
           CD_USE_RECDIM = .FALSE.
           GOTO 1000
        ENDIF

* get the statistics for the given variable
        CALL NCVINQ(cdfid, varid, vname, vartyp, nvdims, vdims,
     .              nvatts,cdfstat)
        IF (cdfstat.NE.NCNOERR) GOTO 5100

* does it use the record axis?
        IF ( vdims(nvdims) .EQ. recdim ) THEN
           CD_USE_RECDIM = .TRUE.
        ELSE
           CD_USE_RECDIM = .FALSE.
        ENDIF

* successful completion
 1000   status = merr_ok
        RETURN

* error exit
 5100   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_USE_RECDIM', cdfid,
     .        no_varid, no_errstring, no_errstring, *5900 )
 5900   RETURN
	END
