        SUBROUTINE BUILD_T_AGG_AXIS(dset_num, nagfiles, memb_filename,
     .             memb_tstep_start, memb_tstep_end, memb_nstep_start,
     .             memb_nsteps, memb_order, status)


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


*  Modify the time axis that was defined by the first file of the T-agg
*  through the inclusion time information from all of the member files
*  in the aggregation.  Fill in the file time metadata while creating
*  the aggregated time axis and assign the ordering of the member files
*  according to this time axis.


* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V695+ 10/15 *sh* -- based upon DEFINE_EF_AGGREGATE_GRID
* V67    5/16 *acm*   Fixes for member sets with single-point time axes having bounds
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V7022 1/17 *acm* ticket 2497, Back off the auto-detection of monthly axes.
* V71   2/17 *acm* ticket 2507, When agg axis matches an existing one, fix swapping 
*                  of axis into grids - the swap axis may be in multiple grids.
* V71   2/17 *acm* ticket 2507, further fix for when several variables have the same grid.
* V72 8/17 *acm*   For trac enhancement #767 -- dynamic coordinate storage
* V74 1/18 *acm*   Issue 1853 correctly handle aggregations of one timestep
* v767 6/22 *acm* See issue 112: Detect whether T dimension becomes too 
*                 large for 4-byte integer indexing
C 12/2022 *acm* Code cleanup: Declare variable tedge as array to avoid rank-mismatch 
*               errors when compiling under gfortran v10.

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

*	dset_num	 - dataset number 1st file -- to become the T agg
*	nagfiles	 - number of files in the aggregation (input)
*	memb_filename	 - array of filenames (input)
*	memb_tstep_start - array of first timestep in each file (output)
*	memb_tstep_end	 - array of last  timestep in each file (output)
*       memb_nstep_start - array of time axis indices of the first step (output)
*	memb_nsteps	 - array of N timesteps in each file (output)
*	memb_order	 - sorted ordering index of files (output)
*	status		 - return status (output)
*

* Argument definitions
        INTEGER dset_num, nagfiles, memb_nstep_start(nagfiles), 
     .          memb_nsteps(nagfiles), memb_order(nagfiles), status
        REAL*8  memb_tstep_start(nagfiles), memb_tstep_end(nagfiles)
        CHARACTER*512 memb_filename(nagfiles)


* local variable declarations:
	LOGICAL TM_DFPEQ_TOL, ITSA_TRUEMONTH_AXIS, need_midpoints, need_T0_adjust,
     .		swap_axes, true_month, memb_true_month
	INTEGER TM_LENSTR1, TM_GET_CALENDAR_ID, TM_FIND_LIKE_DYN_LINE,
     .		memb_tline(nagfiles), taxis, nsteps, dtype, vlen, blen,
     .		i, ifile, istep, ivar, TMAP_status,
     .		cdfid, vid, bounds_id, gg, ilo, ihi, l, imemb,
     .		startpts(2), numpts(2), d1s, d1e, d2s, d2e,
     .		agg_unit_code, file_unit_code, agg_cal_id, pre_exist, 
     .		nsteps_trumon, iline, sumstep, bigline
	REAL	TM_WORLD_AX, T0_ADJUST, GET_LINE_COORD, TM_WW_AXLEN,
     .		SECS_FROM_BC, 
     .		tt, delta, highest_tstep, dfirst_delta, 
     .		depsilon, start, tunit, hdel, tstep, tedge(1), tn, 
     .		memb_sort(nagfiles), axwwlen, bc_to_t0, secsperyear, 
     .		secs2start, timefac, steps_total
	CHARACTER LEFINT*20, lname*20, vname*64, bname*12, units_str*128,
     .		  calendar_str*128, agg_t0_str*20, file_t0_str*20

	PARAMETER (depsilon = 2.**(-42)) 

* initialize
	DO i = 1, nagfiles 
	   memb_tline(i) = unspecified_int4
	ENDDO

* initialize using information from the first file of the aggregation
	taxis          = ds_time_axis(dset_num)
	IF (taxis .EQ. mnormal) GOTO 5050   ! no time axis?
	nsteps         = line_dim(taxis) ! n steps in first file ... will grow
	vname          = line_name_orig(taxis)
	vlen           = TM_LENSTR1(vname)
	dtype          = unspecified_int4  ! irrelevant from older netCD
	need_midpoints = .FALSE.           ! assume cell edges are provided
	agg_unit_code  = line_unit_code(taxis)
	agg_T0_str     = line_T0(taxis)
	agg_cal_id     = TM_GET_CALENDAR_ID (line_cal_name(taxis))

* note that this routine is called under a promise that
* dset_num is the *only* dataset that uses the time axis pointed to by
* ds_time_axis.  Thus our procedure here will be to define the aggregated T
* axis in as a new axis and then replace the pointers to the old axis with the
* new. 

* Was the first file found to have a true-monthly time axis?
* (see old versions of this in SVN and update them, if we want to do true-month axis handling.)

	true_month = .FALSE.

* Put the first datasets timesteps into a new line. We will put the final
* aggregated time axis into the line taxis at the end.

	CALL TM_ALLO_TMP_LINE(iline, status)
	memb_tline(1) = iline
	CALL GET_LINE_DYNMEM (nsteps, iline, status)
	IF (status .NE. ferr_ok) GOTO 5100

        IF ( line_regular(taxis) ) THEN
* ... convert the so-far regular 1st file axis into a forced-irregular axis

	   line_regular(taxis) = .FALSE.
	   
	   DO istep = 1, nsteps
	      CALL PUT_LINE_COORD ( linemem(iline)%ptr, istep, line_start(taxis)
     .		+ (istep-1)* line_delta(taxis) )
	   ENDDO

	   IF (nsteps .EQ. 1) THEN
	      CALL PUT_LINE_COORD ( lineedg(iline)%ptr, 1, 
     .		       line_start(taxis) - line_delta(taxis)/2. )
	      CALL PUT_LINE_COORD ( lineedg(iline)%ptr, 2, 
     .		       line_start(taxis) + line_delta(taxis)/2. )
	   ELSE
	      line_delta(iline) = unspecified_val8
	      need_midpoints = .TRUE.
	   ENDIF
	
        ELSE
	    CALL COPY_LINE_COORDS (linemem(taxis)%ptr, linemem(iline)%ptr, nsteps)
	    CALL COPY_LINE_COORDS (lineedg(taxis)%ptr, lineedg(iline)%ptr, nsteps+1)
	ENDIF

**
* Loop through the files of the aggregation, appending time step values.
* insert the metadata of the first file (so far opened as netCDF dataset)
	   ifile = 1
	   memb_nsteps     (ifile) = nsteps
	   memb_tstep_start(ifile) = GET_LINE_COORD (linemem(iline)%ptr, 1)
	   memb_tstep_end  (ifile) = GET_LINE_COORD (linemem(iline)%ptr, nsteps)

	   steps_total = nsteps
	   IF (steps_total .GT. arbitrary_large_int4) GOTO 5500

	      DO ifile = 2, nagfiles
* ... first read the length of the time axis from the file to be aggregated
	         CALL CD_QUIK_NCOPEN (dset_num, memb_filename(ifile),
     .				      vname(:vlen),
     .				      cdfid, vid, bounds_id,
     .				      units_str, calendar_str,
     .				      nsteps, TMAP_status)
	         IF (TMAP_status .NE. merr_ok) GOTO 5010
* ... check to see if calendar, units, and T0 are consistent with the agg axis
	         CALL CD_GET_T0(memb_filename(ifile), units_str, calendar_str,
     .                  agg_cal_id, file_unit_code, file_t0_str, status)
	         need_T0_adjust = (file_unit_code .NE. agg_unit_code)
     .			     .OR. (file_T0_str    .NE. agg_T0_str)
	         IF (need_t0_adjust) CALL T0_ADJUST_INIT ( agg_cal_id,
     .				          file_unit_code, file_t0_str,
     .					  agg_unit_code,  agg_t0_str, status )
	         memb_nsteps(ifile) = nsteps

		 steps_total = steps_total + nsteps
		 IF (steps_total .GT. arbitrary_large_int4) GOTO 5500

* grab a temporary dynamic axis slot for it
	         CALL TM_ALLO_TMP_LINE(iline, status)
		 memb_tline(ifile) = iline

		 CALL GET_LINE_DYNMEM (nsteps, iline, status)
		 IF (status .NE. ferr_ok) GOTO 5100

* ... read the array of time steps from the file
	         CALL CD_RD_R8_ARR(cdfid, vid,1, nsteps,dtype,vname(:vlen),
     .				   linemem(iline)%ptr, status)
	         IF ( status .NE. merr_ok ) GOTO 5010

* ... adjust the time step values to the agg-chosen "<units> since <T0>"
	         IF (need_T0_adjust) THEN
	            DO istep = 1, nsteps
		       tstep = GET_LINE_COORD (linemem(iline)%ptr, istep)
	               CALL PUT_LINE_COORD ( linemem(iline)%ptr, istep, T0_ADJUST(tstep))
	            ENDDO
	         ENDIF

	         memb_tstep_start(ifile) = GET_LINE_COORD (linemem(iline)%ptr, 1)
	         memb_tstep_end  (ifile) = GET_LINE_COORD (linemem(iline)%ptr, nsteps)

	         IF (bounds_id .EQ. unspecified_int4) THEN
* ... cell boundaries NOT available in this file
*     if not available in any one file, they are effectively missing
	            need_midpoints = .TRUE.

	         ELSEIF (.NOT. need_midpoints) THEN
* ... cell boundaries ARE available in this file
*     store them in high line memory, backwards ordered
	            bname = 'bounds array'
                    blen = 12
*     in CF the bounds are stored 2D Nx2, whereas in Ferret they are simply N+1
*     so two reads are needed
*       read #1) the lower bound array and 
*       read #2) the topmost upper bound (a single point)
*     the cells are *assumed* to be contiguous -- no checking done

*     #1) read the lower box bound values (nsteps values)
                    startpts(1) = 1
                    startpts(2) = 1
                    numpts(1) = 1
                    numpts(2) = nsteps

                    d1s = startpts(1)
                    d1e = numpts(1)
                    d2s = startpts(2)
                    d2e = numpts(2)

                    CALL CD_RD_R8_BNDS( cdfid, bounds_id,
     .			  startpts, numpts, dtype, bname(:blen),
     .                    lineedg(iline)%ptr,  
     .                    d1s, d1e, d2s, d2e, status )
                    IF ( status .NE. merr_ok ) GOTO 5010

*     #2) read the highest upper box bound value (a single value)
                    startpts(1) = 2
                    startpts(2) = nsteps
                    numpts(1) = 1
                    numpts(2) = 1

                    d1s = startpts(1)
                    d1e = numpts(1)
                    d2s = startpts(2)
                    d2e = numpts(2)

                    CALL CD_RD_R8_BNDS( cdfid, bounds_id,
     .			  startpts, numpts, dtype,  bname(:blen),
     .                    tedge,  
     .                    d1s, d1e, d2s, d2e, status )
                    IF ( status .NE. merr_ok ) GOTO 5010
		    CALL PUT_LINE_COORD ( lineedg(iline)%ptr, nsteps+1, tedge(1))

* ... adjust the bounds values to the agg-wide "<units> since <T0>"
	            IF (need_T0_adjust) THEN
	               DO istep = 1, nsteps+1
		          tstep = GET_LINE_COORD (lineedg(iline)%ptr, istep)
	                  CALL PUT_LINE_COORD ( lineedg(iline)%ptr, istep, T0_ADJUST(tstep))
	               ENDDO
	            ENDIF
	         ENDIF
	         CALL CD_CLOSE_SET ( cdfid, status)
	         IF ( status .NE. merr_ok ) GOTO 5010
	      ENDDO    ! end of member file loop

* START: check over and finalize the aggregation time axis

* Get the ordering of the files according to the start time
           nsteps = 0
	   DO i = 1, nagfiles
	      memb_sort(i) = memb_tstep_start(i)
              nsteps = nsteps + memb_nsteps(i)
	   ENDDO
           CALL SORT_TSERIES_DSETS(nagfiles, memb_order, memb_sort)

* combine the coordinates for all lines in order
 
           CALL TM_ALLO_TMP_LINE(bigline, status)
	   CALL GET_LINE_DYNMEM (nsteps, bigline, status)
	   IF (status .NE. ferr_ok) GOTO 5100
	   line_dim(bigline) = nsteps

	   i = 1
	   ifile = memb_order(i)
	   iline = memb_tline(ifile)
           memb_nstep_start(ifile) = 1
	   sumstep = memb_nsteps(ifile)

           CALL COPY_LINE_COORDS (linemem(iline)%ptr, linemem(bigline)%ptr, sumstep)

* Copy the edges, (but not the uppermost one?  Or, as we fill them in, check the upper
* bound from each member set matches or not inconsistent with the lower of the next??
           IF (.NOT. need_midpoints) 
     .        CALL COPY_LINE_COORDS (lineedg(iline)%ptr, lineedg(bigline)%ptr, sumstep)

C check for overlap?
c	   tlast = GET_LINE_COORD (linemem(bigline)%ptr, istep)   ! last time coordinate
c	   elast = GET_LINE_COORD (lineedg(bigline)%ptr, istep+1) ! upper gridcell edge
C check for overlap???

	   DO i = 2, nagfiles
	      ifile = memb_order(i)
	      iline = memb_tline(ifile)
              memb_nstep_start(ifile) = sumstep + 1

	      DO istep = 1, memb_nsteps(ifile)
	         sumstep = sumstep + 1
	         tstep =  GET_LINE_COORD (linemem(iline)%ptr, istep)
	         CALL PUT_LINE_COORD ( linemem(bigline)%ptr, sumstep, tstep)

		 IF (.NOT.need_midpoints) THEN
	            tstep =  GET_LINE_COORD (lineedg(iline)%ptr, istep)
	            CALL PUT_LINE_COORD ( lineedg(bigline)%ptr, sumstep, tstep)
		 ENDIF

	      ENDDO

	   ENDDO

* Uppermost cell edge.

	   IF (.NOT.need_midpoints) THEN
	      tstep =  GET_LINE_COORD (lineedg(iline)%ptr, memb_nsteps(nagfiles)+1)
	      CALL PUT_LINE_COORD ( lineedg(bigline)%ptr, sumstep+1, tstep)
	   ENDIF

* deal with cell boundaries
* ... compute time midpoints if not supplied
	   IF (need_midpoints) 
     .        CALL TM_IRREGAX_MID( linemem(bigline)%ptr, lineedg(bigline)%ptr, nsteps )

* We have now gathered all the coordinates. Re-allocate taxis and put them there,
* so all the grid setup remains as is

	   CALL FREE_LINE_DYNMEM (taxis)
	   CALL GET_LINE_DYNMEM (nsteps, taxis, status)
	   IF (status .NE. ferr_ok) GOTO 5100

           CALL COPY_LINE_COORDS (linemem(bigline)%ptr, linemem(taxis)%ptr, sumstep)
           CALL COPY_LINE_COORDS (lineedg(bigline)%ptr, lineedg(taxis)%ptr, sumstep+1)
	   CALL TM_DEALLO_DYN_LINE(bigline)
	   CALL FREE_LINE_DYNMEM (bigline)

	   line_dim(taxis) = nsteps

* if the coordinates have been saved as irregular we must now
*  check to see if the coords may be regular after all

	   IF (nsteps .GT. 1) dfirst_delta = GET_LINE_COORD (linemem(taxis)%ptr, 2) -
     .           GET_LINE_COORD (linemem(taxis)%ptr, 1)

*  ... first sanity check monotonicity and bounds bracketing
	   IF (nsteps.EQ.1) GOTO 100

	   tt  = GET_LINE_COORD (linemem(taxis)%ptr, 1)
	   tedge(1) = GET_LINE_COORD (lineedg(taxis)%ptr, 1)

* Use this double loop to report errors relative to the member datasets
	   istep = 1
	   DO i = 1, nagfiles
	      ifile = memb_order(i)
	      DO l = 1, memb_nsteps(ifile)  ! timestep within member files
	         IF (i.EQ.1 .AND. l.EQ.1) THEN 
	            CYCLE       ! 1st point on axis		
	         ELSE
	            istep = istep + 1
	            tn = GET_LINE_COORD (linemem(taxis)%ptr, istep)
	            tedge(1) = GET_LINE_COORD (lineedg(taxis)%ptr, istep)
	            IF (tn .LE. tt) GOTO 5300
	            IF (tn .LE. tedge(1)) GOTO 5400
	            tt = tn
		 ENDIF
	      ENDDO
	   ENDDO

	   ifile = memb_order(nagfiles)  ! for error message

*   check regularity of cell positions 

	   DO istep = 1, nsteps-1
	      delta = GET_LINE_COORD (linemem(taxis)%ptr, istep+1) - 
     .		      GET_LINE_COORD (linemem(taxis)%ptr, istep)
	      IF (.NOT.TM_DFPEQ_TOL(dfirst_delta, delta, depsilon) )
     .								GOTO 200
	   ENDDO

*   check regularity of cell bounds
	   DO istep = 1, nsteps-1
	      delta = GET_LINE_COORD (lineedg(taxis)%ptr, istep+1) - 
     .		      GET_LINE_COORD (lineedg(taxis)%ptr, istep)
	      IF (.NOT.TM_DFPEQ_TOL(dfirst_delta, delta, depsilon) )
     .								GOTO 200
	   ENDDO

* It's regular after all!  - change from irregular designation
 100	   CONTINUE
	   line_regular(taxis) = .TRUE.
	   line_start  (taxis) = GET_LINE_COORD (linemem(taxis)%ptr, 1) 
	   IF (nsteps .GT. 1) line_delta(taxis) = dfirst_delta
	   CALL FREE_LINE_DYNMEM (taxis)

* END: check over and finalize the time axis

* does this new time axis match an existing one?
 200	pre_exist = TM_FIND_LIKE_DYN_LINE( taxis )
	swap_axes = pre_exist .NE. unspecified_int4
	IF (swap_axes) THEN
	   ds_time_axis(dset_num) = pre_exist
	   IF (.NOT.line_regular(taxis) ) CALL FREE_LINE_DYNMEM (taxis)
	ENDIF

* correct all time-dependent variables in this dataset
*  ... the upper L limit was set based upon the first file, alone
*  ... and the time axis might just have been swapped for a pre-existing one
*      (ticket 2507, do not wipe out info needed for further grids once the
*       first grid found has its axis swapped.)

	DO ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .EQ. dset_num ) THEN
	      gg  = ds_grid_number(ivar)
	      IF (grid_line(t_dim,gg).EQ.taxis .OR. grid_line(t_dim,gg).EQ.pre_exist) THEN
	         ds_grid_end(t_dim,ivar) = nsteps
	         IF (swap_axes) THEN
* ... was this grid modified in this loop an instant ago?
*     note: Shoud use the dset var LIST!   Super-inefficient nested loop
	            DO i = 1, ivar-1
	               IF ( ds_var_setnum(i) .EQ. dset_num ) THEN
	                  IF (ds_grid_number(i) .EQ. gg) GOTO 500 !already done
	               ENDIF
	            ENDDO
	            grid_line(t_dim,gg) = pre_exist
* ... "use" the pre-existing one; "un-use" the new one
	            CALL TM_USE_LINE(pre_exist)
	            CALL TM_DEALLO_DYN_LINE(taxis)
	            ds_time_axis(dset_num) = pre_exist
	         ENDIF
	      ENDIF
	   ENDIF
 500	   CONTINUE
	ENDDO
	IF (swap_axes) THEN
*          ??? should be FREE_LINE_DYNMEM ??? - but already done above
*	   IF (.NOT.line_regular(taxis)) CALL TM_DEALLO_DYN_LINE(taxis)
	   bigline = pre_exist
	ENDIF

* Is the axis we have created a monthly axis?  IF so store it as such.
* when and if we implement this go back to SVN for the code, and 
* modify it for dynamic-coordinate-memory.

* Should this axis be treated as a modulo axis?
	
	taxis = ds_time_axis(dset_num)
	axwwlen = 0.
        axwwlen = TM_WW_AXLEN(taxis)

	secsperyear = DBLE(cals_yeardays(agg_cal_id)) * (24.D0 * 60.D0 * 60.D0)
	bc_to_t0 = SECS_FROM_BC( line_t0(taxis), agg_cal_id, status)
	CALL TM_WW_AX_1_N(taxis, start, tstep)
	secs2start = bc_to_t0 + start*line_tunit(taxis)
	
* Set the conversion factor for true-month, months to days
	 grid_line(t_dim, dgrid_buff) = taxis
	 CALL TM_MONTH_TIME_FACTOR (dgrid_buff, true_month, timefac)
	IF (true_month) secs2start = secs2start/ timefac

	line_modulo(taxis) = .FALSE.
	line_modulo_len( taxis ) = 0.

	IF ( (secs2start .LE. secsperyear)
     .	 .AND. (axwwlen*line_tunit(taxis) .LE. secsperyear) ) THEN
	   line_modulo(taxis) = .TRUE.
	   line_modulo_len( taxis ) = timefac* secsperyear/line_tunit(taxis)

	ENDIF



* Free the temporary line memory used for member datasets

 5000	CONTINUE	
	DO i = 1, nagfiles
	   IF (memb_tline(i) .NE. unspecified_int4) THEN
	      CALL TM_DEALLO_DYN_LINE(memb_tline(i))
	      CALL FREE_LINE_DYNMEM (memb_tline(i))
	   ENDIF
	ENDDO

 	RETURN
 5010	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )
 5050	CALL ERRMSG( ferr_grid_definition, status,
     .		'first file of aggregation has no time axis '
     .		//memb_filename(1), *5000 )
 5100	CALL ERRMSG(ferr_insuff_memory,status,
     .	        'unable to define T aggregation axis', *5000)
 5300	lname = LEFINT(l, blen)
	CALL ERRMSG( ferr_grid_definition, status,
     .		'time axis is not monotonic at L='
     .		//lname(:blen)//' of file '//memb_filename(ifile), *5000 )
 5400	lname = LEFINT(l, blen)
	CALL ERRMSG( ferr_grid_definition, status,
     .   'Time bounds do not enclose time step coordinate at L='
     .		//lname(:blen)//' of file '//memb_filename(ifile), *5000 )
 5500   CALL ERRMSG(ferr_grid_definition, status,
     .     'Aggregated dimension too long for 4-byte integer indexing: '
     .		//'Long dimensions not implemented', *5000)

	END
