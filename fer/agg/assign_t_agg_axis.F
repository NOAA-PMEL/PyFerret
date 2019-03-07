        SUBROUTINE ASSIGN_T_AGG_AXIS(given_taxis, tmatch, toverlap,
     .         dset_num, nagfiles, memb_filename, memb_tstep_start,
     .         memb_tstep_end, memb_nstep_start, memb_nsteps,
     .         memb_order, status)

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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
*

*  Use the given time axis, after validating, as the aggregation axis.
*  Fill in the file time metadata and assign the ordering of the member
*  files according to this time axis.

        IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xdset_info.cd_equiv'
        include 'xprog_state.cmn'
        include 'xtm_grid.cmn_text'
        include 'xunits.cmn_text'
        include 'errmsg.parm'
        include 'xvariables.cmn'
        include 'xcontext.cmn'
        include 'xfr_grid.cmn'
        include 'interp_stack.parm'
        include 'calendar.decl'
        include 'calendar.cmn'
        include 'netcdf.inc'


* arguments:

*       given_taxis      - time axis number to use (input)
*       tmatch           - method of matching times to given_taxis (input, can be empty);
*                          eg, '@ASN', '@BIN', '@XACT', '@XACT:0.001'
*       toverlap         - allowed number of overlapping time values per datset (input);
*                          negative (such as unspecified_int4) == unlimited
*       dset_num         - dataset number 1st file -- to become the T agg (input)
*       nagfiles         - number of files in the aggregation (input)
*       memb_filename    - array of filenames (input)
*       memb_tstep_start - array of first timestep in each file (output)
*       memb_tstep_end   - array of last timestep in each file (output)
*       memb_nstep_start - array of time axis indices of the first step (output)
*       memb_nsteps      - array of number of timesteps in each file (output)
*       memb_order       - sorted ordering index of files (output)
*       status           - return status (output)
*

* Argument definitions
        INTEGER given_taxis, toverlap, dset_num, nagfiles,
     .          memb_nstep_start(nagfiles), memb_nsteps(nagfiles),
     .          memb_order(nagfiles), status
        REAL*8  memb_tstep_start(nagfiles), memb_tstep_end(nagfiles)
        CHARACTER*512 memb_filename(nagfiles)
        CHARACTER*(*) tmatch

* local variable declarations:
        LOGICAL need_t0_adjust
        INTEGER TM_LENSTR1, TM_GET_CALENDAR_ID,
     .          taxis, agg_nsteps, nsteps, dtype, vlen,
     .          idx, ifile, istep, ivar, iline, plen, flen,
     .          cdfid, vid, bounds_id, gg, numover,
     .          agg_unit_code, file_unit_code, agg_cal_id
        REAL*8  TM_WORLD_AX, T0_ADJUST, GET_LINE_COORD,
     .          depsilon, epsil, aggstep, tstep, lobnd, upbnd,
     .          memb_sort(nagfiles)
        CHARACTER LEFINT*20, vname*64, units_str*128, numbuff*20,
     .            calendar_str*128, agg_t0_str*20, file_t0_str*20

        PARAMETER( depsilon = 2.0**(-42) )

*       Get the time axis name from the first file
*       (assumes rest of the files use the same name for the axis)
        taxis          = ds_time_axis(dset_num)
        IF (taxis .EQ. mnormal) GOTO 5020  ! no time axis?
        vname          = line_name_orig(taxis)
        vlen           = TM_LENSTR1(vname)

        agg_nsteps    = line_dim(given_taxis)
        dtype         = unspecified_int4   ! irrelevant from older netCD
        agg_unit_code = line_unit_code(given_taxis)
        agg_T0_str    = line_T0(given_taxis)
        agg_cal_id    = TM_GET_CALENDAR_ID(line_cal_name(given_taxis))

*       The allowed difference for an exact match (epsil) is in units of the given time axis
        IF ( tmatch .EQ. '@ASN' ) THEN
           GOTO 1000
        ELSE IF ( tmatch .EQ. '@BIN' ) THEN
           epsil = -1.0
        ELSE IF ( tmatch .EQ. '@XACT' ) THEN
           epsil = depsilon
        ELSE IF ( tmatch(1:6) .EQ. '@XACT:' ) THEN
           READ(tmatch(7:), *, ERR=5050) epsil
           IF ( epsil .LT. 0.0 ) GOTO 5050
        ELSE IF ( tmatch .EQ. ' ' ) THEN
           epsil = depsilon
*          Default is to try @XACT; if not successful, try @BIN; if not successful, use @ASN
        ELSE
           GOTO 5050
        ENDIF

        DO ifile = 1, nagfiles
*          Read the length of the time axis from the file to be aggregated
           CALL CD_QUIK_NCOPEN(dset_num, memb_filename(ifile),
     .                         vname(:vlen), cdfid, vid, bounds_id,
     .                         units_str, calendar_str, nsteps, status)
           IF (status .NE. merr_ok) GOTO 5010

*          Check to see if calendar, units, and T0 are consistent with the agg axis
           CALL CD_GET_T0(memb_filename(ifile), units_str,
     .                    calendar_str, agg_cal_id,
     .                    file_unit_code, file_t0_str, status)
           IF (status .NE. merr_ok) THEN
*             Only @ASN can be used
              IF ( tmatch .EQ. ' ' ) GOTO 1000
              GOTO 5010
           ENDIF
           need_t0_adjust = (file_unit_code .NE. agg_unit_code)
     .                 .OR. (file_T0_str .NE. agg_T0_str)
           IF (need_t0_adjust) THEN
              CALL T0_ADJUST_INIT(agg_cal_id, file_unit_code,
     .                file_t0_str, agg_unit_code, agg_t0_str, status)
              IF (status .NE. ferr_ok) THEN
*                Only @ASN can be used
                 IF ( tmatch .EQ. ' ' ) GOTO 1000
                 GOTO 5020
              ENDIF
           ENDIF

*          Grab a temporary dynamic axis slot for the file time axis
           CALL TM_ALLO_TMP_LINE(iline, status)
           CALL GET_LINE_DYNMEM(nsteps, iline, status)
           IF (status .NE. ferr_ok) GOTO 5010

*          Read the array of time steps from the file
           CALL CD_RD_R8_ARR(cdfid, vid, 1, nsteps, dtype,
     .              vname(:vlen), linemem(iline)%ptr, status)
           IF ( status .NE. merr_ok ) THEN
              CALL FREE_LINE_DYNMEM(iline)
              GOTO 5010
           ENDIF

*          Adjust the time step values to the agg-chosen "<units> since <T0>"
           IF ( need_t0_adjust ) THEN
              DO istep = 1, nsteps
                 tstep = GET_LINE_COORD(linemem(iline)%ptr, istep)
                 CALL PUT_LINE_COORD(linemem(iline)%ptr, istep,
     .                               T0_ADJUST(tstep))
              ENDDO
           ENDIF

  100      tstep = GET_LINE_COORD(linemem(iline)%ptr, 1)
           IF ( epsil .GE. 0.0 ) THEN
*             Match the time axis coordinates "exactly", with a tolerance of epsil

              IF ( line_regular(given_taxis) ) THEN
                 DO idx = 1, agg_nsteps
                    aggstep = line_start(given_taxis) +
     .                        (idx-1) * line_delta(given_taxis)
                    IF ( ABS(tstep - aggstep) .LE. epsil ) THEN
                       memb_nstep_start(ifile) = idx
                       GOTO 200
                    ENDIF
                 ENDDO
                 IF ( tmatch .EQ. ' ' ) THEN
*                   Try @BIN; previous file matches still valid
                    epsil = -1.0
                    GOTO 100
                 ENDIF
                 CALL FREE_LINE_DYNMEM(iline)
                 GOTO 5020
  200            DO istep = 2, nsteps
                    tstep = GET_LINE_COORD(linemem(iline)%ptr, istep)
                    idx = idx + 1
                    IF ( idx .GT. agg_nsteps ) THEN
*                      Only @ASN can be used
                       CALL FREE_LINE_DYNMEM(iline)
                       IF ( tmatch .EQ. ' ' ) GOTO 1000
                       GOTO 5040
                    ENDIF
                    aggstep = line_start(given_taxis) +
     .                        (idx-1) * line_delta(given_taxis)
                    IF ( ABS(tstep - aggstep) .GT. epsil ) THEN
                       IF ( tmatch .EQ. ' ' ) THEN
*                         Try @BIN; previous file matches still valid
                          epsil = -1.0
                          GOTO 100
                       ENDIF
                       CALL FREE_LINE_DYNMEM(iline)
                       GOTO 5020
                    ENDIF
                 ENDDO
              ELSE
                 DO idx = 1, agg_nsteps
                    aggstep = GET_LINE_COORD(linemem(given_taxis)%ptr,idx)
                    IF ( ABS(tstep - aggstep) .LE. epsil ) THEN
                       memb_nstep_start(ifile) = idx
                       GOTO 300
                    ENDIF
                 ENDDO
                 IF ( tmatch .EQ. ' ' ) THEN
*                   Try @BIN; previous file matches still valid
                    epsil = -1.0
                    GOTO 100
                 ENDIF
                 CALL FREE_LINE_DYNMEM(iline)
                 GOTO 5020
  300            DO istep = 2, nsteps
                    tstep = GET_LINE_COORD(linemem(iline)%ptr, istep)
                    idx = idx + 1
                    IF ( idx .GT. agg_nsteps ) THEN
*                      Only @ASN can be used
                       CALL FREE_LINE_DYNMEM(iline)
                       IF ( tmatch .EQ. ' ' ) GOTO 1000
                       GOTO 5040
                    ENDIF
                    aggstep = GET_LINE_COORD(linemem(given_taxis)%ptr,idx)
                    IF ( ABS(tstep - aggstep) .GT. epsil ) THEN
                       IF ( tmatch .EQ. ' ' ) THEN
*                         Try @BIN; previous file matches still valid
                          epsil = -1.0
                          GOTO 100
                       ENDIF
                       CALL FREE_LINE_DYNMEM(iline)
                       GOTO 5020
                    ENDIF
                 ENDDO
              ENDIF

           ELSE
*             Match the time axis coordinates by cell boundaries

              IF ( line_regular(given_taxis) ) THEN
                 DO idx = 1, agg_nsteps
                    lobnd = line_start(given_taxis) +
     .                      (idx-1.5) * line_delta(given_taxis)
                    upbnd = line_start(given_taxis) +
     .                      (idx-0.5) * line_delta(given_taxis)
                    IF ( (tstep.GE.lobnd) .AND. (tstep.LT.upbnd) ) THEN
                       memb_nstep_start(ifile) = idx
                       GOTO 400
                    ENDIF
                 ENDDO
*                Only @ASN can be used
                 CALL FREE_LINE_DYNMEM(iline)
                 IF ( tmatch .EQ. ' ' ) GOTO 1000
                 GOTO 5020
  400            DO istep = 2, nsteps
                    tstep = GET_LINE_COORD(linemem(iline)%ptr, istep)
                    idx = idx + 1
                    IF ( idx .GT. agg_nsteps ) THEN
*                      Only @ASN can be used
                       CALL FREE_LINE_DYNMEM(iline)
                       IF ( tmatch .EQ. ' ' ) GOTO 1000
                       GOTO 5040
                    ENDIF
                    lobnd = line_start(given_taxis) +
     .                      (idx-1.5) * line_delta(given_taxis)
                    upbnd = line_start(given_taxis) +
     .                      (idx-0.5) * line_delta(given_taxis)
                    IF ( (tstep.LT.lobnd) .OR. (tstep.GE.upbnd) ) THEN
*                      Only @ASN can be used
                       CALL FREE_LINE_DYNMEM(iline)
                       IF ( tmatch .EQ. ' ' ) GOTO 1000
                       GOTO 5020
                    ENDIF
                 ENDDO
              ELSE
                 DO idx = 1, agg_nsteps
                    lobnd = GET_LINE_COORD(lineedg(given_taxis)%ptr,idx)
                    upbnd = GET_LINE_COORD(lineedg(given_taxis)%ptr,idx+1)
                    IF ( (tstep.GE.lobnd) .AND. (tstep.LT.upbnd) ) THEN
                       memb_nstep_start(ifile) = idx
                       GOTO 500
                    ENDIF
                 ENDDO
*                Only @ASN can be used
                 CALL FREE_LINE_DYNMEM(iline)
                 IF ( tmatch .EQ. ' ' ) GOTO 1000
                 GOTO 5020
  500            DO istep = 2, nsteps
                    tstep = GET_LINE_COORD(linemem(iline)%ptr, istep)
                    idx = idx + 1
                    IF ( idx .GT. agg_nsteps ) THEN
*                      Only @ASN can be used
                       CALL FREE_LINE_DYNMEM(iline)
                       IF ( tmatch .EQ. ' ' ) GOTO 1000
                       GOTO 5040
                    ENDIF
                    lobnd = GET_LINE_COORD(lineedg(given_taxis)%ptr,idx)
                    upbnd = GET_LINE_COORD(lineedg(given_taxis)%ptr,idx+1)
                    IF ( (tstep.LT.lobnd) .OR. (tstep.GE.upbnd) ) THEN
*                      Only @ASN can be used
                       CALL FREE_LINE_DYNMEM(iline)
                       IF ( tmatch .EQ. ' ' ) GOTO 1000
                       GOTO 5020
                    ENDIF
                 ENDDO
              ENDIF

           ENDIF
           CALL FREE_LINE_DYNMEM(iline)

*          Assign the time steps for this file using the given axis coordinates
           idx = memb_nstep_start(ifile)
           IF ( line_regular(given_taxis) ) THEN
              memb_tstep_start(ifile) = line_start(given_taxis) +
     .                   line_delta(given_taxis) * (idx-1)
              memb_tstep_end(ifile) = line_start(given_taxis) +
     .                   line_delta(given_taxis) * (idx+nsteps-1)
           ELSE
              memb_tstep_start(ifile) =
     .             GET_LINE_COORD(linemem(given_taxis)%ptr, idx)
              memb_tstep_end(ifile) =
     .             GET_LINE_COORD(linemem(given_taxis)%ptr, idx+nsteps-1)
           ENDIF
           memb_nsteps(ifile) = nsteps

        ENDDO

*       Get the ordering of the files according to the start time
        DO ifile = 1, nagfiles
           memb_sort(ifile) = memb_tstep_start(ifile)
        ENDDO
        CALL SORT_TSERIES_DSETS(nagfiles, memb_order, memb_sort)

*       Check time overlaps between datasets
        DO ifile = 2, nagfiles
           IF ( memb_tstep_start(memb_order(ifile)) .LE.
     .          memb_tstep_end(memb_order(ifile-1)) ) THEN
              plen = TM_LENSTR1(memb_filename(memb_order(ifile-1)))
              flen = TM_LENSTR1(memb_filename(memb_order(ifile)))
*             Deal with the standard/default of no overlaps allowed
              IF ( toverlap .EQ. 0 ) THEN
                 IF ( tmatch .EQ. ' ' ) GOTO 1000
                 CALL ERRMSG(ferr_aggregate_error, status,
     .                'Time axes overlap for '//
     .                memb_filename(memb_order(ifile-1))(1:plen)//
     .                ' and '//
     .                memb_filename(memb_order(ifile))(1:flen), *5000)
              ENDIF

*             Count the number of overlapping time steps
              numover = 1
              nsteps = memb_nstep_start(memb_order(ifile))
              DO idx = 2, memb_nsteps(memb_order(ifile))
                 nsteps = nsteps + 1
                 IF ( line_regular(given_taxis) ) THEN
                    tstep = line_start(given_taxis) + line_delta(given_taxis) * nsteps
                 ELSE
                    tstep = GET_LINE_COORD(linemem(given_taxis)%ptr, nsteps)
                 ENDIF
                 IF ( tstep .GT. memb_tstep_end(memb_order(ifile-1)) )
     .              GOTO 600
                 numover = numover + 1
              ENDDO
*             If we got here, the prior dataset time axis completely covers 
*             this dataset time axis.  Too problematic; only @ASN can be used.
              IF ( tmatch .EQ. ' ' ) GOTO 1000
              CALL ERRMSG(ferr_aggregate_error, status, 
     .             'Time axis for '//
     .             memb_filename(memb_order(ifile-1))(1:plen)//
     .             'completely covers the time axis for '//
     .             memb_filename(memb_order(ifile))(1:flen), *5000)

  600         IF ( (toverlap .GE. 0) .AND. (toverlap .LT. numover) ) THEN
*                Partial overlap is not acceptable; only @ASN can be used
                 IF ( tmatch .EQ. ' ' ) GOTO 1000
                 numbuff = LEFINT(numover, idx)
                 CALL ERRMSG(ferr_aggregate_error, status,
     .                'Excessive ('//numbuff(1:idx)//') overlaps '//
     .                'in time axes for '//
     .                memb_filename(memb_order(ifile-1))(1:plen)//
     .                ' and '//
     .                memb_filename(memb_order(ifile))(1:flen), *5000)
              ENDIF

*             Check that this truncation does not eliminate the prior dataset (that the
*             prior dataset is not a single time at the start time of this dataset).
              IF ( numover .GE. memb_nsteps(memb_order(ifile-1)) ) THEN
                 IF ( tmatch .EQ. ' ' ) GOTO 1000
                 CALL ERRMSG(ferr_aggregate_error, status, 
     .                'Time axis for '//
     .                memb_filename(memb_order(ifile))(1:flen)//
     .                'completely covers the time axis for '//
     .                memb_filename(memb_order(ifile-1))(1:plen), *5000)
             ENDIF

*             Partial overlap acceptable; truncate the time range of the prior dataset
              idx = memb_nstep_start(memb_order(ifile-1))
              nsteps = memb_nsteps(memb_order(ifile-1)) - numover
              IF ( line_regular(given_taxis) ) THEN
                 memb_tstep_end(memb_order(ifile-1)) = line_start(given_taxis) +
     .                line_delta(given_taxis) * (idx+nsteps-1)
              ELSE
                 memb_tstep_end(memb_order(ifile-1)) =
     .                GET_LINE_COORD(linemem(given_taxis)%ptr, idx+nsteps-1)
              ENDIF
              memb_nsteps(memb_order(ifile-1)) = nsteps

*             Let the user know about the truncation
              numbuff = LEFINT(numover, idx)
              CALL WARN('Ignored last '//numbuff(1:idx)//
     .                  ' timesteps of '//
     .                  memb_filename(memb_order(ifile-1))(1:plen)//
     .                  ' due to overlap with '//
     .                  memb_filename(memb_order(ifile))(1:flen))
           ENDIF
        ENDDO

        IF ( tmatch .EQ. ' ' ) THEN
           IF ( epsil .LT. 0.0 ) THEN
              CALL WARN('Time axis coordinates matched using @BIN')
           ELSE
              CALL WARN('Time axis coordinates matched using @XACT')
           ENDIF
        ENDIF
        GOTO 4000

*       Blindly associate the time steps contiguously onto the given time axis
*       in the order the data files were provided
 1000   istep = 0
        DO ifile = 1, nagfiles
*          Read the length of the time axis from the file to be aggregated
           CALL CD_QUIK_NCOPEN(dset_num, memb_filename(ifile),
     .                         vname(:vlen), cdfid, vid, bounds_id,
     .                         units_str, calendar_str, nsteps, status)
           IF (status .NE. merr_ok) GOTO 5010
           IF ( (istep + nsteps) .GT. agg_nsteps ) GOTO 5040
           memb_nstep_start(ifile) = istep + 1
*          Record times from the given time axis
           IF ( line_regular(given_taxis) ) THEN
              memb_tstep_start(ifile) = line_start(given_taxis) +
     .                   line_delta(given_taxis) * istep
              memb_tstep_end(ifile) = line_start(given_taxis) +
     .                   line_delta(given_taxis) * (istep+nsteps-1)
           ELSE
              memb_tstep_start(ifile) =
     .            GET_LINE_COORD(linemem(given_taxis)%ptr, istep+1)
              memb_tstep_end(ifile) =
     .            GET_LINE_COORD(linemem(given_taxis)%ptr, istep+nsteps)
           ENDIF
           memb_nsteps(ifile) = nsteps
           istep = istep + nsteps
           memb_order(ifile) = ifile
        ENDDO
        IF ( tmatch .EQ. ' ' ) THEN
           CALL WARN('Time axis coordinates matched using @ASN')
        ENDIF

*       Correct all time-dependent variables in this aggregate dataset
 4000   ds_time_axis(dset_num) = given_taxis
        DO ivar = 1, maxvars
           IF ( ds_var_setnum(ivar) .EQ. dset_num ) THEN
              gg = ds_grid_number(ivar)
              IF ( grid_line(t_dim,gg) .EQ. taxis .OR.
     .             grid_line(t_dim,gg) .EQ. given_taxis ) THEN
                 ds_grid_end(t_dim,ivar) = agg_nsteps
              ENDIF
              IF ( grid_line(t_dim,gg) .EQ. taxis ) THEN
*                Was this grid modified in this loop an instant ago?
*                Note: Should use the dset var LIST!   Super-inefficient nested loop
                 DO idx = 1, ivar-1
                    IF ( ds_var_setnum(idx) .EQ. dset_num ) THEN
                       IF (ds_grid_number(idx) .EQ. gg) GOTO 4100
                    ENDIF
                 ENDDO
                 grid_line(t_dim,gg) = given_taxis
*                "use" the given time axis; "un-use" the one from the file
                 CALL TM_USE_LINE(given_taxis)
                 CALL TM_DEALLO_DYN_LINE(taxis)
              ENDIF
           ENDIF
 4100      CONTINUE
        ENDDO
        IF ( .NOT.line_regular(taxis) ) CALL FREE_LINE_DYNMEM(taxis)

        status = ferr_ok
 5000   RETURN

 5010   CALL ERRMSG(ferr_TMAP_error, status, ' ', *5000)
 5020   CALL ERRMSG(ferr_aggregate_error, status,
     .   'Given time axis invalid for the aggregation', *5000)
 5040   CALL ERRMSG(ferr_aggregate_error, status,
     .   'Given time axis not long enough for the aggregation', *5000)
 5050   idx = TM_LENSTR1(tmatch)
        CALL ERRMSG(ferr_aggregate_error, status,
     .   'Invalid time axis assocation "'//tmatch(1:idx)//'"', *5000)

        END
