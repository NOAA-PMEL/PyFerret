	INTEGER FUNCTION CD_ISIT_RECDIM( cdfid, grid, idim, status )

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
