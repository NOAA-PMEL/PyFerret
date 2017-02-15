	SUBROUTINE BUILD_T_AGG_AXIS (dset_num, nagfiles,
     .			memb_order, memb_filename,
     .			memb_tstep_start, memb_tstep_end,
     .			memb_nsteps, t_regular, use_strict, status)


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
*  through the inclusion time information from all of the member files in
*  the aggregation.
*  If the file time metadata information was not provided (i.e. in the
*  argument arrays memb_*), then fill in that metadata here, while creating the
*  aggregated time axis
*  If the files as-provided were out of order, then sort them


* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V695+ 10/15 *sh* -- based upon DEFINE_EF_AGGREGATE_GRID
* V67    5/16 *acm*   Fixes for member sets with single-point time axes having bounds
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V7022 1/17 *acm* ticket 2497, Back off the auto-detection of monthly axes.
* V71   2/17 *acm* ticket 2507, When agg axis identical to an existing one,
*                  fix swapping of axis into grids - the swap axis may be in
*                  multiple grids.

      IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
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
	include 'netcdf.inc'


* arguments:

*	dset_num	- dataset number 1st file -- to become the T agg
*	nagfiles	- number of files in the aggregation (input)
*	memb_order	- ordering index of files (input empty/output filled)
*	memb_filename	- array of filenames (input)
*	memb_tstep_start- array of first timestep in each file (input)
*	memb_tstep_end	- array of last  timestep in each file (input)
*	memb_nsteps	- array of N timesteps in each file (input)
*			  (unspecified_int4 signals no metadata provided)
*	t_regular	- flag to signal promise of regular timesteps (input)
*	use_strict	- flag to allow micro-adjustments of T axis (input)
*	status		- return status (output)
*

* Setting the argument t_regular=.TRUE. signals a promise that every
* individual file will be regularly spaced with the identical delta,
* including the delta between files

* Argument definitions
	LOGICAL		t_regular, use_strict
	INTEGER		nagfiles, dset_num, status
	INTEGER		memb_nsteps(nagfiles), memb_order(nagfiles)
	REAL*8		memb_tstep_start(nagfiles), memb_tstep_end(nagfiles)
	CHARACTER*512	memb_filename(nagfiles)


* local variable declarations:
	LOGICAL TM_DFPEQ_TOL, ITSA_TRUEMONTH_AXIS, need_midpoints, need_T0_adjust,
     .		swap_axes, true_month, memb_true_month
	INTEGER TM_LENSTR1, TM_GET_CALENDAR_ID, TM_FIND_LIKE_DYN_LINE,
     .		taxis, nsteps, max_line_available, dtype, vlen, blen,
     .		i, ifile, istep, ivar, TMAP_status, isubsc1,
     .		cdfid, vid, bounds_id, gg, ilo, ihi, l, imemb,
     .		startpts(2), numpts(2), d1s, d1e, d2s, d2e,
     .		agg_unit_code, file_unit_code, agg_cal_id, pre_exist, 
     .		nsteps_trumon
	REAL	TM_WORLD_AX, T0_ADJUST,
     .		tt, delta, highest_tstep, dfirst_delta, 
     .		depsilon, start, tunit, coord_hi, hdel, coord_new
	CHARACTER LEFINT*20, lname*20, vname*64, bname*12, units_str*128,
     .		  calendar_str*128, agg_t0_str*20, file_t0_str*20

* local parameter declaration
* (this definition should be in a shared declarations file)
	REAL	depsilon_46
	PARAMETER (depsilon_46 = 2.**(-46)) 

* initialize using information from the first file of the aggregation
	taxis          = ds_time_axis(dset_num)
	IF (taxis .EQ. mnormal) GOTO 5050   ! no time axis?
	nsteps         = line_dim(taxis) ! n steps in first file ... will grow
	vname          = line_name_orig(taxis)
	vlen           = TM_LENSTR1(vname)
	max_line_available = maxlinestore
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
c	true_month = ITSA_TRUEMONTH_AXIS (taxis)
	true_month = .FALSE.

c	IF (true_month .AND. .NOT.t_regular) THEN
c
c* Loop through the files of the aggregation, checking for more compatible true-month axes.
c* The time axis from member 1 is not in line_mem, so just keep reading into the next slots
c
c	   nsteps_trumon = nsteps
c	   isubsc1 = next_line_mem_pos
c	   
c	   coord_hi = line_start(taxis) + (nsteps-1)* line_delta(taxis)
c
c* Translate fake true-month timestep to actual month-length step.
c
c	   CALL TM_TCOORD_TRUMONTH (taxis, coord_hi,  
c     .		       box_middle, coord_new)
c
c	   memb_order(1) = 1
c	   memb_nsteps(1) = nsteps
c	   memb_tstep_start(1) = line_start(taxis) 
c	   memb_tstep_end  (1) = coord_new
c
c	   TMLOOP: DO ifile = 2, nagfiles
c* ... first read the length of the time axis from the file to be aggregated
c	      CALL CD_QUIK_NCOPEN (dset_num, memb_filename(ifile),
c     .				      vname(:vlen),
c     .				      cdfid, vid, bounds_id,
c     .				      units_str, calendar_str,
c     .				      nsteps, TMAP_status)
c	      IF (TMAP_status .NE. merr_ok) GOTO 5010
c* ... check to see if calendar, units, and T0 are consistent with the agg axis
c	      CALL CD_GET_T0(memb_filename(ifile), units_str, calendar_str,
c     .                  agg_cal_id, file_unit_code, file_t0_str, status)
c
c	      IF (next_line_mem_pos+nsteps .GT. max_line_available) GOTO 5100
c
c* ... read the array of time steps from the file and read or compute bounds
c	      CALL CD_RD_R8_ARR(cdfid, vid,1, nsteps,dtype,vname(:vlen),
c     .				   line_mem(isubsc1), status)
c	      IF ( status .NE. merr_ok ) GOTO 5010
c
c	      IF (bounds_id .NE. unspecified_int4) THEN
c
c	         startpts(1) = 1
c	         startpts(2) = 1
c	         numpts(1) = 1
c	         numpts(2) = nsteps
c
c	         d1s = startpts(1)
c	         d1e = numpts(1)
c	         d2s = startpts(2)
c	         d2e = numpts(2)
c		 bname = 'bounds array'
c		 blen = 12
c
c		 CALL CD_RD_R8_BNDS( cdfid, bounds_id,
c     .			  startpts, numpts, dtype, bname(:blen),
c     .                    line_mem(isubsc1+nsteps),  
c     .                    d1s, d1e, d2s, d2e, status )
c	         IF ( status .NE. merr_ok ) GOTO 5010
c
c	      ELSE
c	         CALL TM_IRREGAX_MID( isubsc1,isubsc1 )
c	      ENDIF
c
c	      CALL TM_CHECK_MONTHLY_AXIS (line_mem(isubsc1), nsteps, agg_cal_id, 
c     .                                   file_T0_str, file_unit_code, units_str,
c     .                                   start, delta, tunit, memb_true_month)
c
c* Also bail on true-month if the units or t0 are different.
c
c	      need_T0_adjust = (file_unit_code .NE. agg_unit_code)
c     .			     .OR. (file_T0_str    .NE. agg_T0_str)
c	      IF (need_T0_adjust) memb_true_month = .FALSE.
c
c! check value of "start" relative to the stored axis -- checking sorting.
c	      IF (start .LT. coord_hi) memb_true_month = .FALSE.
c	      IF (start-coord_hi .GT. 2.*line_delta(taxis))  memb_true_month = .FALSE.
c	      IF (delta .NE. line_delta(taxis)) memb_true_month = .FALSE.
c
c	      IF (.NOT.memb_true_month) EXIT TMLOOP ! from loop
c
c	      nsteps_trumon = nsteps_trumon + nsteps
c	      coord_hi = line_start(taxis) + (nsteps_trumon-1)* line_delta(taxis)
c
c* Translate fake true-month timestep to actual month-length step.
c
c	      CALL TM_TCOORD_TRUMONTH (taxis, coord_hi,  
c     .		       box_middle, coord_new)
c
c	      memb_order(ifile) = ifile
c	      memb_nsteps(ifile) = nsteps
c	      memb_tstep_start(ifile) = start
c	      memb_tstep_end(ifile) = coord_new
c
c	      line_dim(taxis) = nsteps_trumon
c	      t_regular = .TRUE.
c	  
c	   ENDDO TMLOOP
c
c* ... If some month isn't a true-mnth axis, then we will not use a true-month axis.
c*     Get the original coordinates from the first member set and continue.
c
c	   IF (true_month .AND. .NOT.memb_true_month) THEN
c	   
c	      t_regular = .FALSE.
c	      ifile = 1
c	      CALL CD_QUIK_NCOPEN (dset_num, memb_filename(ifile),
c     .				      vname(:vlen),
c     .				      cdfid, vid, bounds_id,
c     .				      units_str, calendar_str,
c     .				      nsteps, TMAP_status)
c	      IF (TMAP_status .NE. merr_ok) GOTO 5010
c* ... check to see if calendar, units, and T0 are consistent with the agg axis
c	      CALL CD_GET_T0(memb_filename(ifile), units_str, calendar_str,
c     .                  agg_cal_id, file_unit_code, file_t0_str, status)
c
c* ... read the array of time steps from the file
c	      CALL CD_RD_R8_ARR(cdfid, vid,1, nsteps,dtype,vname(:vlen),
c     .				   line_mem(isubsc1), status)
c	      line_regular(taxis) = .FALSE.
c	      line_unit_code(taxis) = file_unit_code
c	      agg_unit_code = file_unit_code
c	      line_delta(taxis) = unspecified_val8
c
c	      IF (bounds_id .NE. unspecified_int4) THEN
c
c	         startpts(1) = 1
c	         startpts(2) = 1
c	         numpts(1) = 1
c	         numpts(2) = nsteps
c
c	         d1s = startpts(1)
c	         d1e = numpts(1)
c	         d2s = startpts(2)
c	         d2e = numpts(2)
c		 bname = 'bounds array'
c		 blen = 12
c
c		 CALL CD_RD_R8_BNDS( cdfid, bounds_id,
c     .			  startpts, numpts, dtype, bname(:blen),
c     .                    line_mem(isubsc1+nsteps),  
c     .                    d1s, d1e, d2s, d2e, status )
c	      ELSE
c	         CALL TM_IRREGAX_MID( isubsc1,isubsc1 )
c	      ENDIF
c
c	      true_month = .FALSE.
c	   ENDIF
c
c	ENDIF  ! true_month

* Do some of the member sets need bounds?  If so then get them for all sets.
c	IF (.NOT.t_regular .AND. line_regular(taxis)) THEN
c	   need_midpoints = .TRUE.
c	   DO ifile = 1, nagfiles
c	      CALL CD_QUIK_NCOPEN (dset_num, memb_filename(ifile),
c     .				      vname(:vlen),
c     .				      cdfid, vid, bounds_id,
c     .				      units_str, calendar_str,
c     .				      nsteps, TMAP_status)
c	      IF (bounds_id .NE. unspecified_int4) need_midpoints = .FALSE.
c
c	   ENDDO
c	   IF (.NOT. need_midpoints) THEN  ! fill in coordinats and bounds for first member set
c	      hdel = line_delta(taxis)/2.
c	      DO istep = 0, nsteps-1
c	         line_mem(next_line_mem_pos+istep) = line_start(taxis)
c     .		+ istep*line_delta(taxis)
c	         line_mem(next_line_mem_pos+nsteps+istep) = 
c     .		    line_mem(next_line_mem_pos+istep) - hdel
c	         line_mem(next_line_mem_pos+nsteps+istep+1) = 
c     .		    line_mem(next_line_mem_pos+istep) + hdel
c	      ENDDO
c	   ENDIF
c	ENDIF

* set up for a regular or irregular T-agg axis as requested by t_regular
	IF (t_regular .AND. .NOT.line_regular(taxis)) THEN
	   CALL WARN('/T_REGULAR NOT YET IMPLEMENTED')
! T_REGULAR HAS NEVER BEEN TESTED
!* ... convert the irregular t axis of 1st file into a forced-regular axis
!	   line_start(taxis) = line_mem(line_subsc1(taxis))
!	   line_regular(taxis) = .TRUE.
!* !! TODO: WE SHOULD REALLY GARBAGE COLLECT THE ABANDONED SPACE OCCUPIED BY
!*          THE IRREGULAR COORDINATES OF THE FIRST FILE IN THE T-AGG
!*          SEE PACK_LINE_STORAGE
!* quik garb collect if T-axis happens to be the last one in memory
!	   IF (next_line_mem_pos .EQ. line_subsc1(taxis) + nsteps) THEN
!	      next_line_mem_pos = line_subsc1(taxis) ! ouch! orphan some coords
!	      line_subsc1(taxis) =  unspecified_int4
!	   ENDIF
	ELSEIF (.NOT.t_regular .AND. line_regular(taxis)) THEN
* ... convert the so-far regular 1st file axis into a forced-irregular axis
	   IF (next_line_mem_pos+2*line_dim(taxis)+1
     .					.GT.maxlinestore) GOTO 5100
	   line_regular(taxis) = .FALSE.
	   line_subsc1(taxis) = next_line_mem_pos
	   DO istep = 0, nsteps-1
	      line_mem(next_line_mem_pos+istep) = line_start(taxis)
     .		+ istep*line_delta(taxis)
	   ENDDO
	   line_delta(taxis) = unspecified_val8
	   need_midpoints = .TRUE.
	ELSEIF (.NOT.t_regular .AND. .NOT.line_regular(taxis)) THEN
* ... make sure this axis is the last one currently stored in line_mem
	   IF (next_line_mem_pos .GT. line_subsc1(taxis) + 2*nsteps + 1)
     .								THEN
* !! TODO: WE SHOULD REALLY GARBAGE COLLECT THE ABANDONED SPACE OCCUPIED BY
*          THE ORIGINAL IRREGULAR COORDINATES OF THE FIRST FILE IN THE T-AGG
*          SEE PACK_LINE_STORAGE
* ... copy the time steps and bounds from the first file to the top
	      DO istep = 0, 2*nsteps
	         line_mem(next_line_mem_pos+istep)
     .	                            = line_mem(line_subsc1(taxis)+istep)
	      ENDDO
	      line_subsc1(taxis) = next_line_mem_pos
	   ENDIF
* ... copy the cell boundaries to high memory in reversed order
	   max_line_available = max_line_available - nsteps
	   ilo = line_subsc1(taxis) + nsteps
	   ihi = max_line_available + nsteps
	   DO istep = 0, nsteps
	      line_mem(ihi) = line_mem(ilo)
	      ilo = ilo+1
	      ihi = ihi-1
	   ENDDO

	ENDIF

**************
* Loop through the files of the aggregation, appending time step values.
* Different logic applies if the caller passed in the time axis metadata.
* Because the length of the aggregated T axis is not in general known until
* all of the aggregated files have been consulted, we cannot put the cell
* boundaries in their usual place immediately following the cell positions.
* Instead we temporarily locate the irregular cell boundaries at the *top*
* of the line_mem array working backwards, copying them into their proper
* position after the cell marker coordinates are all in place.

	IF (memb_nsteps(1) .NE. unspecified_int4) THEN

!!!!! THIS LOGIC HAS NOT BEEN TESTED -- INTENDED FOR PYTHON INTERFACE

***
* time metadata has been provided by caller (i.e. python program)
* note that the metadata of the first file is ignored since already read
* directly from the file, itself
	   IF (t_regular) THEN
	      CONTINUE
!	      highest_tstep = line_start(taxis)
!     .			    + line_dim(taxis)*line_delta(taxis)
!	      DO ifile = 2, nagfiles
!	         highest_tstep = memb_tstep_end(ifile)
!	         line_dim(taxis) = line_dim(taxis) + memb_nsteps(ifile)
!	      ENDDO
	   ELSE
	      DO ifile = 2, nagfiles
	         line_dim(taxis) = line_dim(taxis) + memb_nsteps(ifile)
	         IF (next_line_mem_pos+line_dim(taxis)+memb_nsteps(ifile)
     .			.GT.maxlinestore)			GOTO 5100
	         IF (memb_nsteps(ifile) .GT. 1 ) THEN
	            delta = (memb_tstep_end(ifile) - memb_tstep_start(ifile))
     .		       / (memb_nsteps(ifile)-1)
	         ELSE
	            delta = 0.0
	         ENDIF
	         DO istep = 0, memb_nsteps(ifile)-1
	            line_mem(next_line_mem_pos+line_dim(taxis)+istep) 
     .		         = memb_tstep_start(ifile) + delta*istep
	         ENDDO
	         line_dim(taxis) = line_dim(taxis) + memb_nsteps(ifile)
	      ENDDO
	      need_midpoints = .TRUE.
	   ENDIF
	ELSE


***
* time metadata has not been provided - we must read it from the files
* insert the metadata of the first file (so far opened as netCDF dataset)
	   memb_nsteps     (1) = nsteps
	   memb_tstep_start(1) = TM_WORLD_AX(1,     taxis,box_middle) 
	   memb_tstep_end  (1) = TM_WORLD_AX(nsteps,taxis,box_middle) 

	   IF (t_regular) THEN
	      CALL WARN('/T_REGULAR NOT YET IMPLEMENTED')
!	      highest_tstep = line_start(taxis)
!     .			    + line_dim(taxis)*line_delta(taxis)
!	      DO ifile = 2, nagfiles
!* ... read the length of the time axis from the file to be aggregated
!	         CALL CD_QUIK_NCOPEN (dset_num, memb_filename(ifile),
!     .				      vname(:vlen),
!     .				      cdfid, vid, bounds_id,
!     .				      units_str, calendar_str,
!     .				      nsteps, TMAP_status)
!	         IF (TMAP_status .NE. merr_ok) GOTO 5010
!* user gives /T_REGULAR as a promise of file compatibility, so no checking
!* of units and calendar attributes
!	         memb_nsteps(ifile) = nsteps
!* ... read the first and last time step values from the axis
!! it's faster to compute the first and last than to read them
!! reading last, anyway, so that if the promise of regular points is a lie
!! at least the range of the time axis will be correct
!!	         CALL CD_RD_R8_1(cdfid, vid,     1,dtype,vname(:vlen),
!!     .                           memb_tstep_start(ifile), status )
!!	         IF ( status .NE. merr_ok ) GOTO 5010
!	         CALL CD_RD_R8_1(cdfid, vid,nsteps,dtype,vname(:vlen),
!     .                           memb_tstep_end(ifile),   status )
!	         IF ( status .NE. merr_ok ) GOTO 5010	         
!	         line_dim(taxis) = line_dim(taxis) + nsteps
!	         highest_tstep = memb_tstep_end(ifile)
!	         line_delta(taxis) = (highest_tstep - line_start(taxis))
!     .				   /(line_dim(taxis) - 1)
!	         memb_tstep_start(ifile) = highest_tstep
!     .				       - (nsteps-1)*line_delta(taxis)
!	         CALL CD_CLOSE_SET ( cdfid, status)
!	         IF ( status .NE. merr_ok ) GOTO 5010
!	      ENDDO
	   ELSE
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
	         max_line_available = max_line_available-(nsteps+1) ! n+1 edges
	         IF (next_line_mem_pos+line_dim(taxis)+nsteps
     .			.GT.max_line_available)			GOTO 5100
* ... read the array of time steps from the file
	         isubsc1 = line_subsc1(taxis)+line_dim(taxis)
	         CALL CD_RD_R8_ARR(cdfid, vid,1, nsteps,dtype,vname(:vlen),
     .				   line_mem(isubsc1), status)
	         IF ( status .NE. merr_ok ) GOTO 5010



* ... adjust the time step values to the agg-chosen "<units> since <T0>"
	         IF (need_T0_adjust) THEN
	            DO istep = isubsc1, isubsc1+nsteps-1
	               line_mem(istep) = T0_ADJUST(line_mem(istep))
	            ENDDO
	         ENDIF

	         line_dim(taxis) = line_dim(taxis) + nsteps
	         memb_tstep_start(ifile) = line_mem(isubsc1)
	         memb_tstep_end  (ifile) = line_mem(isubsc1+nsteps-1)

	         IF (bounds_id .EQ. unspecified_int4) THEN
* ... cell boundaries NOT available in this file
*     if not available in any one file, they aare effectively missing
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
     .                    line_mem(max_line_available),  
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
     .                    line_mem(max_line_available+nsteps),  
     .                    d1s, d1e, d2s, d2e, status )
                    IF ( status .NE. merr_ok ) GOTO 5010

* ... reorder the newest nstep+1 bounds values into backwards order,
*     ensuring one pass through the loop for single-step axis
	            DO istep = 0, MAX(0,nsteps/2-1)

	               ilo = max_line_available+istep
	               ihi = max_line_available+nsteps-istep
	               tt = line_mem(ilo)
	               line_mem(ilo) = line_mem(ihi)
	               line_mem(ihi) = tt
	            ENDDO

* ... adjust the bounds values to the agg-wide "<units> since <T0>"
	            IF (need_T0_adjust) THEN
	               DO istep = max_line_available, max_line_available+nsteps
	                  line_mem(istep) = T0_ADJUST(line_mem(istep))
	               ENDDO
	            ENDIF
	         ENDIF
	         CALL CD_CLOSE_SET ( cdfid, status)
	         IF ( status .NE. merr_ok ) GOTO 5010
	      ENDDO    ! end of member file loop
	   ENDIF
	ENDIF
	nsteps = line_dim(taxis)   ! length of aggregated axis

****** START: check over and finalize the aggregation time axis
	IF (.NOT.t_regular) THEN

	   IF (.NOT.need_midpoints) THEN
* explicit cell boundaries have been stored in high line_mem indices
* note that instead of nsteps+1 edges, at this point we have nsteps+nagfiles
* because each member file has contributed both a low and a hgh edge
* we have held onto the extra points because the member order may be re-sorted
*   1. reorder the cell bounds back to forwards order
*   2. move the cell bounds from temporary storage to their normal position

* ... 1. reorder the bounds to forward order
	      ilo = max_line_available
	      ihi = ilo + nsteps + nagfiles - 1
	      DO istep = 0, (nsteps+nagfiles)/2
	         tt = line_mem(ilo)
	         line_mem(ilo) = line_mem(ihi)
	         line_mem(ihi) = tt
	         ilo = ilo + 1
	         ihi = ihi - 1
	         IF (ihi .LT. ilo) EXIT  ! exit loop; 1-point axes casuses this
	      ENDDO

* ... 2. shift the bounds down to sit just after the coordinates
	      ilo = line_subsc1(taxis) + nsteps
	      ihi = max_line_available
	      DO istep = 0, nsteps+nagfiles-1
	         line_mem(ilo) = line_mem(ihi)
	         line_mem(ihi) = bad_val4        ! for good house-keeping
	         ilo = ilo + 1
	         ihi = ihi + 1
	      ENDDO
	   ENDIF

* if the fileas as provided were out of order, then sort them here
* sort the time coordinates and bounds in line_mem to match the sort order
	   CALL SORT_TSERIES_DSETS(nagfiles, nsteps, need_midpoints, 
     .				   memb_nsteps, memb_order,
     .				   line_mem(line_subsc1(taxis)),
     .				   line_mem(line_subsc1(taxis)+nsteps) )

* deal with cell boundaries
	   next_line_mem_pos =  next_line_mem_pos + 2*nsteps + 1

* ... compute time midpoints if not supplied
	   IF (need_midpoints) THEN
	      IF (line_subsc1(taxis)+2*nsteps+1.GT.maxlinestore) GOTO 5100
	      CALL TM_IRREGAX_MID(line_subsc1(taxis), nsteps)

	   ELSE

* remove the duplicated cell edges: top of file k duplicates bottom of file k+1
*    istep counts time steps in the aggregated time axis 
*    ifile counts the member files
*    imemb counts the time step in the member current file

	      ilo = line_subsc1(taxis) + nsteps
	      i = 1
	      ifile = memb_order(i)
	      imemb = memb_nsteps(ifile) ! start at top of first file
	      istep = imemb
	      DO WHILE (istep .LE. nsteps)
	         istep = istep + 1
	         imemb = imemb + 1
	         IF (imemb .GT. memb_nsteps(ifile)) THEN
*     ... start of next member file
*     ... skipping the top cell edge of each file
	            i = i + 1
	            IF (i .LE. nagfiles) ifile = memb_order(i)
	            imemb = 1
	         ENDIF
	         l = ilo + istep
	         line_mem(l) = line_mem(l+i-1)
	      ENDDO
* ... copy (dont skip) the upper edge of the aggregate axis
	      line_mem(l+1) = line_mem(l+i)
	   ENDIF

* if the coordinates have been saved as irregular we must now
*  check to see if the coords may be regular after all

*  ... first sanity check monotonicity and bounds bracketing
	   IF (nsteps.EQ.1) GOTO 100
	   ilo = line_subsc1(taxis)    ! L step on agg taxis
	   ihi = ilo + nsteps - 1
           dfirst_delta = line_mem(ilo+1) - line_mem(ilo)
           depsilon = depsilon_46 * 2.*(ABS(line_mem(ilo))
     .		    / dfirst_delta )
	   DO i = 1, nagfiles
	      ifile = memb_order(i)
	      DO l = 1, memb_nsteps(ifile)  ! timestep within memer files
	         IF (i.EQ.1 .AND. l.EQ.1) THEN 
	            CYCLE       ! 1st point on axis		
	         ELSE
	            ilo = ilo + 1
	            tt  = line_mem(ilo)
	            IF (tt .LE. line_mem(ilo-1))	 GOTO 5300
	            IF (tt .LT. line_mem(ilo+nsteps)) GOTO 5400 ! lo bnd chk
	         ENDIF
	      ENDDO
	   ENDDO
	   ifile = memb_order(nagfiles)  ! for error message
	   IF (tt .GT. line_mem(ilo+nsteps+1))	 GOTO 5400 ! hi bnd chk

*   check regularity of cell positions 
	   ilo = line_subsc1(taxis)
	   DO istep = 1, nsteps-1
	      ilo = ilo + 1
	      delta = line_mem(ilo) - line_mem(ilo-1)
	      IF (.NOT.TM_DFPEQ_TOL(dfirst_delta, delta, depsilon) )
     .								GOTO 200
	   ENDDO

*   check regularity of cell bounds
	   ilo = line_subsc1(taxis) + nsteps   ! lower cell bound of 1st pt
	   DO istep = ilo, ilo+nsteps-1	
	      delta = line_mem(istep+1) - line_mem(istep)
	      IF (.NOT. TM_DFPEQ_TOL(dfirst_delta, delta, depsilon) )
     .								GOTO 200
	   ENDDO

* It's regular after all!  - change from irregular designation
 100	   CONTINUE
	   ilo = line_subsc1(taxis)
	   ihi = ilo + nsteps - 1
	   line_regular(taxis) = .TRUE.
	   line_start  (taxis) = line_mem(ilo)
	   IF (nsteps .GT. 1) THEN   
	      line_delta(taxis) = (line_mem(ihi)-line_mem(ilo))/(nsteps-1)
	   ELSE
	      line_delta(taxis) = 1.0   ! arbitrary
	   ENDIF
	   line_subsc1 (taxis) = unspecified_int4
	   next_line_mem_pos =  next_line_mem_pos - (2*nsteps + 1) ! restore
	ENDIF
****** END: check over and finalize the time axis

* does this new time axis match an existing one?
 200	pre_exist = TM_FIND_LIKE_DYN_LINE( taxis )
	swap_axes = pre_exist .NE. unspecified_int4
	IF (swap_axes) ds_time_axis(dset_num) = pre_exist

* correct all time-dependent variables in this dataset
*  ... the upper L limit was set based upon the first file, alone
*  ... and the time axis might just have been swapped for a pre-existing one
*      (ticket 2507, do not wipe out info needed for further grids once the
*       first grid found has its axis swapped.)

	DO ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .EQ. dset_num ) THEN
	      gg  = ds_grid_number(ivar)
	      IF (grid_line(t_dim,gg) .EQ. taxis) THEN
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
	IF (swap_axes) taxis = pre_exist

* Is the axis we have created a monthly axis?  IF so store it as such.

c	IF (.NOT. true_month .AND. .NOT.line_regular(taxis)) THEN
c	   CALL TM_CHECK_MONTHLY_AXIS (line_mem( line_subsc1(taxis)), line_dim(taxis), agg_cal_id, 
c     .                                   agg_T0_str, agg_unit_code, units_str,
c     .                                   start, delta, tunit, true_month)
c           IF (true_month) THEN
c              line_regular(taxis) = .TRUE.
c              line_start  (taxis) = start
c              line_delta  (taxis) = delta
c              line_subsc1 (taxis) = unspecified_int4
c	      line_unit_code (taxis) = agg_unit_code
c	      line_units (taxis) = units_str
c              line_tunit  (taxis) = un_convert(agg_unit_code)
c	   ENDIF
c	ENDIF
c

 5000	RETURN
 5010	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )
 5050	CALL ERRMSG( ferr_grid_definition, status,
     .		'first file of aggregation has no time axis '
     .		//memb_filename(1), *5000 )
 5100	CALL ERRMSG(ferr_insuff_memory,status,
     .	        'unable to define T aggregation axis', *5000)
 5200	CALL ERRMSG(ferr_internal, status, 'define_ef_aggregate_grid!',
     .		 *5000)
 5300	lname = LEFINT(l, blen)
	CALL ERRMSG( ferr_grid_definition, status,
     .		'time axis is not monotonic at L='
     .		//lname(:blen)//' of file '//memb_filename(ifile), *5000 )
 5400	lname = LEFINT(l, blen)
	CALL ERRMSG( ferr_grid_definition, status,
     .   'Time bounds do not enclose time step coordinate at L='
     .		//lname(:blen)//' of file '//memb_filename(ifile), *5000 )

	END
