	SUBROUTINE CD_REC_SYNCH ( cdfid, lo, hi, tmrecax, grid,
     .                            start, axid, vartyp, status )

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
* Determine where along the record axis to write data in a netCDF
* output file
* Extend the record axis if the coordinates given fit into the
* axis ordering
* Check carefully (and laboriously!) to determina if axis point spacing is
* regular

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
* note: commented out references to line 1000 are to support disordered axes
* 10/92: bug fix: use TM_FPEQ to check time words
*        performance: write time data outside to avoid multiple ncendf calls
* 12/92: pass back vartyp so data type is correct in cd_write_var axis appends
* 1/97:  issues a one-time warning if an irregular time axis is extended
*	 (since the box bounds will not be saved)
* 6/97:  changed "STATIC" to SAVE for f90 compatibility
*	 also moved var assignments off of declaration line *kob*
* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     lo      - lowest index from variable grid on record axis
*     hi      - highest index from variable grid on record axis
*     tmrecax - axis number (1-4) of record axis in grid (for errmsg, only)
*     grid    - defining grid for variable
*     start   - file record number for next write (returned)
*     axid    - CDF var no of coordinate variable for record axis

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
        include 'tmap_format/gt_lib.parm'
        include 'tmap_format/cd_lib.parm'
        include 'tmap_format/tmap_dims.parm'		! for warning 1/97
        include 'tmap_format/tmap_errors.parm'
        include 'tmap_format/xio.cmn_text'
        include 'tmap_format/xtm_grid.cmn_text'		! for warning 1/97
        external xio_data

#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
        INCLUDE 'TMAP_FORMAT:CD_LIB.PARM'
        INCLUDE 'TMAP_FORMAT/TMAP_DIMS.PARM'		! for warning 1/97
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'		! for warning 1/97
        EXTERNAL XIO_DATA
#endif

* argument declarations
      INTEGER cdfid, lo, hi, tmrecax, start, axid, grid, vartyp, status

* local variable declarations:
      LOGICAL CD_GET_ATTRIB, TM_FPEQ, got_it, regular,
     .        do_warn
      PARAMETER ( do_warn = .TRUE. )
      INTEGER TM_LENSTR1, STR_CASE_BLIND_COMPARE, NCVID,
     .        alen, look, hi_look, lo_look,
     .        ndims, nvars, cdfstat, i100, hilim, i,
     .        nvdim, vdims(6), nvatts,
     .        ngatts, npts, recdim
      CHARACTER axnam*24, buff*16
      CHARACTER*1     dimnam(4)
      DATA dimnam/'I','J','K','L'/
      CHARACTER*13 TM_STRING
!  REAL   r100(100)   ! removed 10/92   
      REAL  delta
      REAL*8 TM_WORLD, new, hi_val, lo_val, val

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* local parameter declarations
      LOGICAL     append
      INTEGER     str_eq
      REAL        unspecified_val4
      PARAMETER ( str_eq = 0,
     .            unspecified_val4 = -2E34,
     .            append = .TRUE. )

* statis variable causes warning to be issued only once
	LOGICAL first_warning
	SAVE	first_warning
	DATA	first_warning/.TRUE./

* initialize
        delta = unspecified_val4

* get the vital statistics for the CDF file
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )

* get the name of the record axis
        CALL NCDINQ( cdfid, recdim, axnam, npts, cdfstat )
        alen = TM_LENSTR1( axnam )

* is there a coordinate variable by this name ?  If not, error (for now)
        axid = NCVID( cdfid, axnam(:alen), cdfstat )
        IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* get the vital statistics of the coordinate variable
        CALL NCVINQ(cdfid, axid, buff, vartyp, nvdim, vdims,
     .                nvatts, cdfstat)

! ... 10/92: ncendf called in CD_WRITE_VAR
!* leave define mode so we can read and write the coordinates
!        CALL NCENDF(cdfid, cdfstat)

* locate the new coordinate values at end of coordinate axis ?
* see if the first of the new coordinates exceeds the last already in the file
        new = TM_WORLD( lo, grid, tmrecax, box_middle )
        IF ( npts .GT. 0 ) THEN
           CALL CD_RD_R8_1( cdfid, axid, npts, vartyp,
     .                      'axis: '//axnam(:alen), hi_val, cdfstat )
           IF ( cdfstat .NE. merr_ok ) GOTO 5900
           CALL CD_RD_R8_1( cdfid, axid, 1, vartyp,
     .                      'axis: '//axnam(:alen), lo_val, cdfstat )
           IF ( cdfstat.NE.merr_ok ) GOTO 5900
* ... (determine the delta of the axis prior to writing new coordinates)
           IF ( npts .GE.2 ) delta = (hi_val - lo_val) / (npts - 1)
           IF ( TM_FPEQ(SNGL(new), SNGL(hi_val)) ) THEN
              start = npts
              GOTO 2000
           ELSEIF ( new .GT. hi_val ) THEN
              start = npts + 1
              GOTO 500		! warning to be issued (1/97)
!              GOTO 2000
           ENDIF
        ELSE
           hi_val = unspecified_val8
           start = 1
           GOTO 500		! warning to be issued (1/97)
!           GOTO 2000
        ENDIF

* binary search to locate new coordinate in file
* ... initialize search with coordinate of first point
        IF ( TM_FPEQ(SNGL(new), SNGL(lo_val)) ) THEN
           start = 1
           GOTO 2000
        ELSEIF ( new .LT. lo_val ) THEN
           GOTO 5200 
        ENDIF
        lo_look = 0
        hi_look = npts + 1
        look = ( lo_look + hi_look ) / 2

* top of search loop
 100    CALL CD_RD_R8_1( cdfid, axid, look, vartyp,
     .                   'axis: '//axnam(:alen), val, cdfstat )
        IF ( cdfstat.NE.merr_ok ) GOTO 5900
        IF ( TM_FPEQ(SNGL(new), SNGL(val)) ) THEN
* ... found it
           start = look
           GOTO 1000
        ELSEIF ( hi_look - lo_look .LE. 1 ) THEN
* ... value not on axis
           GOTO 5200
        ELSEIF ( new .GT. val ) THEN
* ... look higher
           IF ( look .EQ. hi_look ) GOTO 5200
           lo_look = look
           lo_val = val
           look = ( look + hi_look + 1 ) / 2
        ELSE
* ... look lower
           IF ( look .EQ. lo_look ) GOTO 5200
           hi_look = look
           hi_val = val
           look = ( look + lo_look + 1 ) / 2
        ENDIF
        GOTO 100

* issue a one-time warning if we are extending an irregular axis
 500	IF  ( .NOT.line_regular(grid_line(tmrecax,grid))
     .	.AND. first_warning ) THEN
	   CALL TM_NOTE(
     .		'Irregular T axis box limits cannot be saved',
     .		tooplu)
	   CALL TM_NOTE(
     .		'in NetCDF files created by this means.',
     .		tooplu)
	   CALL TM_NOTE(
     .		'To avoid this create the file using the qualifiers',
     .		tooplu)
	   CALL TM_NOTE(
     .		'    /RIGID/HEADING=ENHANCED/LLIMITS=lo:hi',
     .		tooplu)
	   CALL TM_NOTE(
     .		'This warning will not be repeated.',
     .		tooplu)
	   first_warning = .FALSE.
	ENDIF

 1000   CONTINUE
 2000   CONTINUE
!!!! 10/92 - check for axis regularity removed as a part of performance
!!!! enhancement to consolidate all writes under a single ncendf (slow) call

!* this new data will overlay some of the previously existing axis
!* initialize to determine if new points conform to old spacing
! 1000   IF ( delta .EQ. unspecified_val4 ) THEN
!           hi_val = unspecified_val8  ! was only 1 point & we're jumping on it
!        ELSE
!           hi_val = new - delta   ! as if concatenating to the previous point
!        ENDIF
!
!* add coordinates to file (possibly over-writing identical old ones)
!* note: no check is made that just because th first new coordinate matches
!* the successive ones match, too
!* (buffer 100 at a time)
!* also, check for regular point spacing in the coordinates being written
! 2000   regular = .TRUE.
!        DO 2110 i100 = lo, hi, 100
!           hilim = MIN(i100+99, hi)
!           DO 2100 i = i100, hilim
! 2100      r100(i-i100+1) = TM_WORLD( i, grid, tmrecax, box_middle )
!           CALL NCVPT(cdfid, axid, i100-lo+start, hilim-i100+1,
!     .                r100, cdfstat)
!           IF ( cdfstat .NE. NCNOERR ) GOTO 5100
!* ... check for matching regularity of new points as we write
!           DO 2105 i = 1, hilim-i100+1
!              IF ( delta .NE. unspecified_val4 ) THEN
!                 IF ( .NOT.TM_FPEQ(delta,
!     .                     SNGL(r100(i)-hi_val) ) ) regular=.FALSE.
!              ELSEIF ( hi_val .NE. unspecified_val8 ) THEN
!                 delta  = r100(i) - hi_val
!              ENDIF
!              hi_val = r100(i)
! 2105      CONTINUE
! 2110   CONTINUE
!        CALL NCREDF( cdfid, cdfstat ) ! back into define mode
!
!* check the existing point_spacing attribute
!        got_it = CD_GET_ATTRIB( cdfid, axid, 'point_spacing',
!     .                          do_warn, axnam(:alen), buff, 16 )
!        IF ( .NOT.got_it )  GOTO 4000    ! axis not created by FERRET
!        IF ( str_eq .NE. STR_CASE_BLIND_COMPARE(buff,'even') )
!     .                      GOTO 4000    ! once uneven, always uneven
!
!* add the point_spacing attribute
!        IF ( regular ) THEN
!           CALL NCAPTC( cdfid, axid, 'point_spacing', NCCHAR, 5,
!     .              'even', cdfstat )
!        ELSE
!           CALL NCAPTC( cdfid, axid, 'point_spacing', NCCHAR, 7,
!     .              'uneven', cdfstat )
!        ENDIF

* successful completion
 4000   status = merr_ok
        RETURN

* error exits
 5100  CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_REC_SYNCH', cdfid, axid,
     .       'axis: '//axnam(:alen), no_errstring, *5900 )
 5200  CALL TM_ERRMSG
     .      ( merr_badlinedef, status, 'CD_REC_SYNCH', no_descfile,
     .        no_stepfile,
     .       'disordered output coordinate value:'
     .       //TM_STRING(new)//' Axis: '//axnam(:alen),
     .       no_errstring , *5900 )
 5900  CALL NCREDF( cdfid, cdfstat ) ! back into define mode
       RETURN
       END
