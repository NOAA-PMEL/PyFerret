	LOGICAL FUNCTION CD_USE_RECDIM( cdfid, varid, status )

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
