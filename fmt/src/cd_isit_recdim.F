	INTEGER FUNCTION CD_ISIT_RECDIM( cdfid, grid, idim, status )

* Is the indicated is the indicated axis of the grid the netCDF record axis?
* Or can it never become the record axis ?
* Or are both it and the record axis as-yet undefined?

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 6/92

* argument definitions:
*       cdfid   - returned pointer to open CDF file
*       grid    - TMAP grid number
*       idim    - axis number (1-4)
*	status	- return status

* argument declarations:
	INTEGER	      cdfid, grid, idim, status

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
        include 'tmap_format/tmap_errors.parm'
        include 'tmap_format/cd_lib.parm'
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:CD_LIB.PARM'
#endif

* local variable declarations:
      LOGICAL this_ax_exists, recax_exists
      INTEGER NCDID, 
     .        ndims, nvars, ngatts, recdim, cdfstat, nlen, dimid
      CHARACTER CD_AXIS_NAME*32, name*32

* get the axis name
        name = CD_AXIS_NAME( grid, idim, nlen )

* is there a dimension by this name?
        dimid = NCDID( cdfid, name(:nlen), cdfstat )
        this_ax_exists = cdfstat .EQ. NCNOERR

* get vital statistics of file
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )
!        IF ( cdfstat .NE. NCNOERR ) GOTO 5100
        recax_exists = recdim .NE. -1

* so what's the answer?
        IF ( .NOT.(this_ax_exists .OR. recax_exists) ) THEN
           CD_ISIT_RECDIM = pcdf_undefined    ! both undefined as yet
        ELSEIF ( this_ax_exists
     .     .AND. recax_exists
     .     .AND. (dimid .EQ. recdim) ) THEN
           CD_ISIT_RECDIM = pcdf_yes          ! both defined and equal
        ELSE
           CD_ISIT_RECDIM = pcdf_no           ! at least one exists unequal
        ENDIF

* successful completion
 1000   status = merr_ok
        RETURN

* error exit
!5100   CALL TM_ERRMSG
!    .     ( cdfstat+pcdferr, status, 'CD_USE_RECDIM', cdfid,
!    .        no_varid, no_errstring, no_errstring, *5900 )
!5900   RETURN
	END
