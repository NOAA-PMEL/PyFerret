	SUBROUTINE CD_REC_SYNCH ( cdfid, lo, hi, tmrecax, grid,
     .                            start, axid, vartyp, reset, 
     .                            old_box_hi, new_box_lo, do_bounds,
     .                            status )

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

* V570  *acm* 5/04 Do not issue the warning about irregular record axis,
*                  Bounds are used rather than edges for defining irregular 
*                  axis cell boundaries.  Check whether there is a gap
*                  between the high box-bound of the existing axis and
*                  the low box-bound of the new axis to be appended.

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     lo      - lowest index from variable grid on record axis
*     hi      - highest index from variable grid on record axis
*     tmrecax - axis number (1-4) of record axis in grid (for errmsg, only)
*     grid    - defining grid for variable
*     start   - file record number for next write (returned)
*     axid    - CDF var no of coordinate variable for record axis

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
#include "gt_lib.parm"
        include 'cd_lib.parm'
        include 'tmap_errors.parm'
        include 'xio.cmn_text'
        external xio_data

* argument declarations
      LOGICAL do_bounds
      INTEGER cdfid, lo, hi, tmrecax, start, axid, grid, vartyp, 
     .        reset, status

* local variable declarations:
      LOGICAL CD_GET_ATTRIB, TM_DFPEQ, got_bnds
      INTEGER TM_LENSTR1, NCVID,
     .        alen, look, hi_look, lo_look,
     .        ndims, nvars, cdfstat,
     .        nvdim, vdims(6), nvatts,
     .        ngatts, npts, recdim, blen, bndid,
     .        startpts(2), numpts(2)
      CHARACTER axnam*128, buff*16, bname*30
      CHARACTER*1     dimnam(4)
      DATA dimnam/'I','J','K','L'/
      CHARACTER*13 TM_STRING

      REAL*8 TM_WORLD, new_coord, hi_val, lo_val, val,
     .       new_box_lo, old_box_hi, hi_val_1

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* local parameter declarations
      LOGICAL     append, do_warn
      INTEGER     str_eq
      REAL        unspecified_val4
      PARAMETER ( str_eq = 0,
     .            unspecified_val4 = -2E34,
     .            append = .TRUE., do_warn=.TRUE. )

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
        new_coord = TM_WORLD( lo, grid, tmrecax, box_middle )
        new_box_lo = 0.D0
        old_box_hi = 0.D0
        reset = 0

        IF ( npts .GT. 0 ) THEN
           CALL CD_RD_R8_1( cdfid, axid, npts, vartyp,
     .                      'axis: '//axnam(:alen), hi_val, cdfstat )
           IF ( cdfstat .NE. merr_ok ) GOTO 5900
           CALL CD_RD_R8_1( cdfid, axid, 1, vartyp,
     .                      'axis: '//axnam(:alen), lo_val, cdfstat )
           IF ( cdfstat.NE.merr_ok ) GOTO 5900
* 4/99 *kob* - call to new routine tm_dfpeq for double precision compares
           IF ( TM_DFPEQ(new_coord,hi_val) ) THEN
              start = npts
              GOTO 4000
           ELSEIF ( new_coord .GT. hi_val ) THEN
              start = npts + 1
              GOTO 1000
           ENDIF
        ELSE
           hi_val = unspecified_val8
           start = 1
           GOTO 4000
        ENDIF

* binary search to locate new coordinate in file
* ... initialize search with coordinate of first point
        IF ( TM_DFPEQ(new_coord, lo_val) ) THEN
           start = 1
           GOTO 4000
        ELSEIF ( new_coord .LT. lo_val ) THEN
           GOTO 5200 
        ENDIF
        lo_look = 0
        hi_look = npts + 1
        look = ( lo_look + hi_look ) / 2

* top of search loop
 100    CALL CD_RD_R8_1( cdfid, axid, look, vartyp,
     .                   'axis: '//axnam(:alen), val, cdfstat )
        IF ( cdfstat.NE.merr_ok ) GOTO 5900
        IF ( TM_DFPEQ(new_coord, val) ) THEN
* ... found it
           start = look
           GOTO 1000
        ELSEIF ( hi_look - lo_look .LE. 1 ) THEN
* ... value not on axis
           GOTO 5200
        ELSEIF ( new_coord .GT. val ) THEN
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

 1000   CONTINUE

* is there a bounds attribute for this axis? 

       got_bnds = CD_GET_ATTRIB( cdfid, axid, 'bounds',
     .                       do_warn, axnam(:alen), bname, 30 )

       IF (got_bnds) THEN

* Always write bounds when appending to axis which has bounds

          do_bounds = .TRUE.  

* Get upper bound of last box on existing axis

          blen = TM_LENSTR1(bname)
          bndid = NCVID( cdfid, bname(:blen), cdfstat )

          startpts(1) = 2  ! get uppper bound, 1 point
          numpts(1) = 1

* get upper bound of last point to be used on existing axis
          startpts(2) = start-1  
          numpts(2) = 1

          CALL CD_RD_R8_BNDS( cdfid, bndid, startpts, numpts, vartyp, 
     .                    bname(:blen), old_box_hi, status )

* Get lower bound of first point to be output

          new_box_lo = TM_WORLD( lo, grid, tmrecax, box_lo_lim )

* Reconcile upper bound of last point on existing axis with lower bound of new axis
* reset = 0  no change to bounds is needed
*       = 1  gap between old_hi and new_lo, so add a void point with these as bounds
*       = 2  need to reset new_lo to match old_hi

          IF (TM_DFPEQ(old_box_hi, new_box_lo) ) THEN 
*  no need to adjust; bounds line up.
              reset = 0

          ELSE IF (old_box_hi .LT. new_box_lo) THEN
*  need to add a void point, with bounds [old_box_hi , new_box_lo]
              reset = 1

          ELSE   !  (old_box_hi .GT. new_box_lo) 
             IF (old_box_hi .LT. new_coord) THEN 
*  old_box_hi is between new_box_lo and new_coord, so use old_box_hi for new_box_lo
                reset = 2
                new_box_lo = old_box_hi

             ELSE IF (old_box_hi .GE. new_coord) THEN
                IF (new_box_lo .GT. hi_val) THEN 
*  Issue an error message: high bound of last point of existing axis greater than
*  the first coordinate of points being appended; cannot reset old_box_hi
                   GO TO 5300
                ELSE   
*  new_box_lo is < hi_val, so use old_box_hi for new_box_lo
                   reset = 2
                   new_box_lo = old_box_hi
                ENDIF
             ENDIF
          ENDIF

       ELSE   ! (not got_bnds)

* Check whether we are outputting the next point on a regularly spaced axis.
* If we are skipping points, issue a warning.

          IF (new_coord .GT. hi_val) THEN  

              CALL CD_RD_R8_1( cdfid, axid, npts-1, vartyp,
     .              'axis: '//axnam(:alen), hi_val_1, cdfstat )
              IF (.NOT. TM_DFPEQ(hi_val-hi_val_1, new_coord-hi_val) ) THEN
                 CALL WARN( 
     . 'Appending to NetCDF record axis which has no bounds attribute.')
                 CALL WARN( 
     . 'This will result in incorrect box sizes on record axis: '
     .             // axnam(:alen))
                 CALL WARN( 
     . 'Write the data initially with the /BOUNDS qualifier')
              ENDIF
           ENDIF

       ENDIF

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
     .       //TM_STRING(new_coord)//' Axis: '//axnam(:alen),
     .       no_errstring , *5900 )
 5300  CALL TM_ERRMSG     
     .      ( merr_badlinedef, status, 'CD_REC_SYNCH', no_descfile,
     .        no_stepfile,
     . 'output coordinate is inside upper bound of existing axis cell:'
     .       //TM_STRING(new_coord)//' Axis: '//axnam(:alen),
     .       no_errstring , *5900 )
 5900  CALL NCREDF( cdfid, cdfstat ) ! back into define mode
       RETURN
       END
