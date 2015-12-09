	SUBROUTINE BUILD_T_AGG_AXIS (dset_num, nagfiles, memb_filename,
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


* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V695+ 10/15 *sh* -- based upon DEFINE_EF_AGGREGATE_GRID

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
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'
	include 'interp_stack.parm'
	include 'netcdf.inc'


* arguments:

*	dset_num	- dataset number 1st file -- to become the T agg
*	nagfiles	- number of files in the aggregation (input)
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
	INTEGER		memb_nsteps(nagfiles)
	REAL*8		memb_tstep_start(nagfiles), memb_tstep_end(nagfiles)
	CHARACTER*128	memb_filename(nagfiles)


* local variable declarations:
	LOGICAL TM_DFPEQ_TOL, need_midpoints, need_T0_adjust
	INTEGER TM_LENSTR1, TM_GET_CALENDAR_ID,
     .		taxis, nsteps, max_line_available, dtype, vlen, blen,
     .		ifile, istep, ivar, TMAP_status, isubsc1,
     .		cdfid, vid, bounds_id, gg, i, ilo, ihi, l,
     .		startpts(2), numpts(2), d1s, d1e, d2s, d2e,
     .		agg_unit_code, file_unit_code, agg_cal_id
	REAL	TM_WORLD_AX, T0_ADJUST,
     .		tt, delta, highest_tstep, dfirst_delta, 
     .		depsilon
	CHARACTER LEFINT*20, lname*20, vname*64, bname*12, units_str*128,
     .		  calendar_str*128, agg_t0_str*20, file_t0_str*20

* local parameter declaration
* (this definition should be in a shared declarations file)
	REAL	depsilon_46
	PARAMETER (depsilon_46 = 2.**(-46)) 

* this code will be needed in a future version:
* logic from DEFINE_EF_AGGREGATE_GRID to allow differing T0 dates
!	      CALL TDEST_WORLD_INIT (gnum2, grid, t_dim, status)
!	      IF (status .NE. ferr_ok) RETURN 
!	      DO i2 = 1, nT
!	         line_mem(ipt) = TDEST_WORLD(i2, gnum2, t_dim, box_middle)
!	         line_mem(ipt+npts) =
!     .				 TDEST_WORLD(i2, gnum2, t_dim, box_lo_lim)
!	         ipt = ipt + 1
!	      ENDDO
!	   ENDDO
!	   line_mem(ipt1+2*npts) =
!     .				 TDEST_WORLD(nT, gnum2, t_dim, box_hi_lim)


* initialize using information from the first file of the aggregation
	taxis          = ds_time_axis(dset_num)
	IF (taxis .EQ. mnormal) GOTO 5050   ! no time axis?
	nsteps         = line_dim(taxis) ! n steps in first file ... will grow
	vname          = line_name_orig(taxis)
	vlen           = TM_LENSTR1(vname)
	max_line_available = maxlinestore
	dtype          = unspecified_int4  ! irrelevant from older netCD
	need_midpoints = .FALSE.
	agg_unit_code  = line_unit_code(taxis)
	agg_T0_str     = line_T0(taxis)
	agg_cal_id     = TM_GET_CALENDAR_ID (line_cal_name(taxis))

* note that when this routine is called there is an implicit promise that
* dset_num is the *only* dataset that uses the time axis pointed to by
* ds_time_axis.  Thus our procedure here will be to define the aggregated T
* axis in as a new axis and then replace the pointers to the old axis with the
* new. 


* set up for a regular or irregular T-agg axis as requested by t_regular
	IF (t_regular .AND. .NOT.line_regular(taxis)) THEN
* ... convert the irregular t axis of 1st file into a forced-regular axis
	   line_start(taxis) = line_mem(line_subsc1(taxis))
	   line_regular(taxis) = .TRUE.
* !! TODO: WE SHOULD REALLY GARBAGE COLLECT THE ABANDONED SPACE OCCUPIED BY
*          THE IRREGULAR COORDINATES OF THE FIRST FILE IN THE T-AGG
*          SEE PACK_LINE_STORAGE
* quik garb collect if T-axis happens to be the last one in memory
	   IF (next_line_mem_pos .EQ. line_subsc1(taxis) + nsteps) THEN
	      next_line_mem_pos = line_subsc1(taxis) ! ouch! orphan some coords
	      line_subsc1(taxis) =  unspecified_int4
	   ENDIF
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

***
* time metadata has been provided by caller (i.e. python program)
* note that the metadata of the first file is ignored since already read
* directly from the file, itself
	   IF (t_regular) THEN
	      highest_tstep = line_start(taxis)
     .			    + line_dim(taxis)*line_delta(taxis)
	      DO ifile = 2, nagfiles
	         highest_tstep = memb_tstep_end(ifile)
	         line_dim(taxis) = line_dim(taxis) + memb_nsteps(ifile)
	      ENDDO
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
	      highest_tstep = line_start(taxis)
     .			    + line_dim(taxis)*line_delta(taxis)
	      DO ifile = 2, nagfiles
* ... read the length of the time axis from the file to be aggregated
	         CALL CD_QUIK_NCOPEN (dset_num, memb_filename(ifile),
     .				      vname(:vlen),
     .				      cdfid, vid, bounds_id,
     .				      units_str, calendar_str,
     .				      nsteps, TMAP_status)
	         IF (TMAP_status .NE. merr_ok) GOTO 5010
* user gives /T_REGULAR as a promise of file compatibility, so no checking
* of units and calendar attributes
	         memb_nsteps(ifile) = nsteps
* ... read the first and last time step values from the axis
! it's faster to compute the first and last than to read them
! reading last, anyway, so that if the promise of regular points is a lie
! at least the range of the time axis will be correct
!	         CALL CD_RD_R8_1(cdfid, vid,     1,dtype,vname(:vlen),
!     .                           memb_tstep_start(ifile), status )
!	         IF ( status .NE. merr_ok ) GOTO 5010
	         CALL CD_RD_R8_1(cdfid, vid,nsteps,dtype,vname(:vlen),
     .                           memb_tstep_end(ifile),   status )
	         IF ( status .NE. merr_ok ) GOTO 5010	         
	         line_dim(taxis) = line_dim(taxis) + nsteps
	         highest_tstep = memb_tstep_end(ifile)
	         line_delta(taxis) = (highest_tstep - line_start(taxis))
     .				   /(line_dim(taxis) - 1)
	         memb_tstep_start(ifile) = highest_tstep
     .				       - (nsteps-1)*line_delta(taxis)
	         CALL CD_CLOSE_SET ( cdfid, status)
	         IF ( status .NE. merr_ok ) GOTO 5010
	      ENDDO
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
	         max_line_available = max_line_available-nsteps  ! keep n of em
	         IF (next_line_mem_pos+line_dim(taxis)+nsteps
     .			.GT.max_line_available)			GOTO 5100
* ... read the array of time steps from the file
	         isubsc1 = line_subsc1(taxis)+line_dim(taxis)
	         CALL CD_RD_R8_ARR(cdfid, vid,1, nsteps,dtype,vname(:vlen),
     .				   line_mem(isubsc1), status)
	         IF ( status .NE. merr_ok ) GOTO 5010

* ... adjust the time step values to the agg-wide "<units> since <T0>"
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

*     #2) read the highest upper box bound value (1 value)
*         Note that the upper bound of file K and the lower bound of
*         file K+1 are presumed the same value.  If not we will clobber the 
*	  upper bound og file K in this step **without detecting the problem**
*         We could add code to issue a warning if it seemed worth the trouble.
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

* ... reorder the newest nstep+1 bounds values into backwards order
	            DO istep = 0, nsteps/2-1
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
	      ENDDO
	   ENDIF
	ENDIF

* ToDo - here we could re-order the files and times, if they were out of order
* use values of memb_tstep_start to determine ordering
* Note that the handling of bounds values would need to be changed slightly
* The top bound of file K is the bottom bound of file K+1 if correctly ordered
* by default. Currently we over-write the redundant value. We would need to
* postpone removal of this redundancy (store N+1 bounds for each file) until
* after the sorting of files has been completed.
	CONTINUE


*************  clean up the axis ***************
* either flesh out the forced-regular axis
	nsteps = line_dim(taxis)   ! length of aggregated axis
	IF (t_regular) THEN
	   CONTINUE   ! line_delta kept up to date in loop above
	
	ELSE
* or deal with cell boundaries
	   next_line_mem_pos =  next_line_mem_pos + 2*nsteps + 1

* ... compute time midpoints if not supplied
	   IF (need_midpoints) THEN
	      IF (line_subsc1(taxis)+2*nsteps+1.GT.maxlinestore) GOTO 5100
	      CALL TM_IRREGAX_MID(line_subsc1(taxis), nsteps)

	   ELSE

* ... explicit cell boundaries have been stored in high line_mem indices
*   1. reorder the cell bounds back to forwards order
*   2. move the cell bounds from thei temporary storage

* ... 1. reorder the bounds to forward order
	      ilo = max_line_available
	      ihi = ilo + nsteps
	      DO istep = 0, nsteps/2
	         tt = line_mem(ilo)
	         line_mem(ilo) = line_mem(ihi)
	         line_mem(ihi) = tt
	         ilo = ilo + 1
	         ihi = ihi - 1
	      ENDDO
* ... 2. shift the bounds down to sit just after the coordinates
	      ilo = line_subsc1(taxis) + nsteps
	      ihi = max_line_available
	      DO istep = 0, nsteps
	         line_mem(ilo) = line_mem(ihi)
	         line_mem(ihi) = bad_val4        ! for good house-keeping
	         ilo = ilo + 1
	         ihi = ihi + 1
	      ENDDO
	   ENDIF

* if the coordinates have been saved as irregular we must now
*  check to see if the coords may be regular after all

*  ... first sanity check monotonicity and bounds bracketing
	   ilo = line_subsc1(taxis)    ! L step on agg taxis
!	   ihi = ilo + nsteps - 1
           dfirst_delta = line_mem(ilo+1) - line_mem(ilo)
           depsilon = depsilon_46 * 2.*(ABS(line_mem(ilo))
     .		    / dfirst_delta )
	   DO ifile = 1, nagfiles
	      DO l = 1, memb_nsteps(ifile)  ! timestep within individual file
	         IF (ifile.EQ.1 .AND. l.EQ.1) THEN 
	            CYCLE       ! 1st point on axis		
	         ELSE
	            ilo = ilo + 1
	            tt  = line_mem(ilo)
	            IF (tt .LE. line_mem(ilo-1))	GOTO 5300
	            IF (tt .LT. line_mem(ilo+nsteps))	GOTO 5400 ! lo bnd chk 
	         ENDIF
	      ENDDO
	   ENDDO
	   IF (tt .GT. line_mem(ilo+nsteps+1))		GOTO 5400 ! hi bnd chk

*   check regularity of cell positions 
	   ilo = line_subsc1(taxis)
	   DO istep = 1, nsteps-1
	      ilo = ilo + 1
	      delta = line_mem(ilo) - line_mem(ilo-1)
	      IF (.NOT.TM_DFPEQ_TOL(dfirst_delta, delta, depsilon) )
     .								GOTO 100
	   ENDDO

*   check regularity of cell bounds
	   ilo = line_subsc1(taxis) + nsteps   ! lower cell bound of 1st point
	   DO i = ilo, ilo+nsteps-1	
	      delta = line_mem(i+1) - line_mem(i)
	      IF (.NOT. TM_DFPEQ_TOL(dfirst_delta, delta, depsilon) )
     .								GOTO 100
	   ENDDO

* It's regular after all!  - change from irregular designation
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
 

* correct the upper L limit for all time-dependent variables in this dataset
* as the current L limit was set based upon the first file, alone
 100	DO ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .EQ. dset_num ) THEN
	      gg  = ds_grid_number(ivar)
	      IF (grid_line(t_dim,gg) .EQ. taxis) THEN
	         ds_grid_end(t_dim,ivar) = nsteps
	      ENDIF
	   ENDIF
	ENDDO

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
