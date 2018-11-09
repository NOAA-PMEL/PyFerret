	SUBROUTINE SORT_FMRC_DSETS (nsets_in, member_sets, maxmembers,
     .				    nsets_out, nt_out, status)
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
* NOAA/PMEL, Seattle, WA - TMAP 

* sort the given list of datasets based upon the starting time steps
* and insert dummy dataset where the forecast sequence has gaps
*
* V698 *sh* 11/15 
*            2/16 -- allow missing datasets among nsets_in
*            3/16 -- bug fix in epsilon used to screen bad gaps

* calling arguments
*   nsets_in		- number of datasets given to be aggregated (input)
*   member_sets		- array of open dataset numbers (input-output)
*   maxmembers		- maximum number of members allowes (input)
*   nsets_out		- number of datasets including dummy fillers (output)
*   nt_out		- time ax len (in case inconsistent) (output)
*   status		- return staus (output)

        IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'netcdf.inc'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'
	include 'xdiag_ctrl.cmn_text'

* calling argument declarations:
	INTEGER nsets_in, nsets_out, nt_out,
     .		maxmembers, member_sets(maxmembers)

* local variable declarations:
	LOGICAL alteredDelta
	INTEGER	ivar, grid, grid1, iset, dset, dset1, taxis, taxis1,
     .		status, nadded, ngap, i, nsets, itmp_memb(maxmembers),
     .		deltaFactor
	REAL TDEST_WORLD, TM_WORLD, d, epsilon, gap, deltaT, deltaT_0,
     .		start_time(maxmembers), rindices(maxmembers),
     .		d_ratio, d_ratio_int
	CHARACTER LEFINT*8

* initialize
	dset1  = member_sets(1)
	taxis1 = ds_time_axis(dset1)
	nt_out = line_dim(taxis1)
	status = ferr_ok
	IF (nsets_in .LE. 1) RETURN

* since forecast series must be regular, and the code below
* is able to detect gaps in the list of forecasts given,we
* will simply remove the user-specified gaps, knowing they
* will be re-inserted automatically
	ngap = 0
	iset = 1
	DO WHILE (iset+ngap .LT. nsets_in)
	   IF (member_sets(iset) .EQ. pdset_dummy) THEN
	      ngap = ngap + 1
	   ELSE
	      iset = iset + 1
	   ENDIF
	   member_sets(iset) = member_sets(iset+ngap)
	ENDDO
	nsets = nsets_in - ngap
	nsets_out = nsets

* locate a variable in the first dataset that uses this time axis
	DO ivar = 1, maxvars
           IF ( ds_var_setnum(ivar) .NE. dset1 ) CYCLE
	   grid1 = ds_grid_number(ivar)
	   IF (grid_line(t_dim,grid1) .EQ. taxis1) GOTO 100
	ENDDO
	GOTO 5100

* loop through the datasets extracting the start time from each one
* Apply the T0 encoding of the first dataset to all
 100	start_time(1) = TM_WORLD( 1, grid1, t_dim, box_middle )
	DO iset = 2, nsets
	   dset  = member_sets(iset)
	   taxis = ds_time_axis(dset)
           IF (line_name_orig(taxis) .NE. line_name_orig(taxis1)) GOTO 5100
	   
* ... length of T axis in the aggregation will be the MIN of all members
	   IF (line_dim(taxis) .LT. nt_out) nt_out = line_dim(taxis)

* ... locate a variable in the first dataset that uses this time axis
	   DO ivar = 1, maxvars
              IF ( ds_var_setnum(ivar) .NE. dset ) CYCLE
	      grid = ds_grid_number(ivar)
	      IF (grid_line(t_dim,grid) .EQ. taxis) GOTO 200
	   ENDDO
	   GOTO 5100

* ... save the start time of this dataset
 200	   CALL TDEST_WORLD_INIT (grid, grid1, t_dim, status)
	   IF (status .NE. ferr_ok) GOTO 5100 
	   start_time(iset) = TDEST_WORLD(1, grid, t_dim, box_middle)
	ENDDO

* sort the dates and the dataset list
* ... put indices into REAL array
	DO iset = 1, nsets
	   rindices(iset) = iset
	ENDDO
	CALL HEAP2(start_time, rindices, unspecified_val8, nsets)
* ... reorder the dataset numbers to match the sort
	DO iset = 1, nsets
	   itmp_memb(iset) = member_sets(INT(rindices(iset)))
	ENDDO
* ... copy the sorted list back into the input array
	DO iset = 1, nsets
	   member_sets(iset) = itmp_memb(iset)
	ENDDO

* determine the intended delta-T between the (ordered) forecasts
* ... first get the minimum delta-T
	deltaT_0 = start_time(2) - start_time(1)
	DO iset = 2, nsets
	   d = start_time(iset)-start_time(iset-1)
	   IF ( d .LT. deltaT_0) deltaT_0 = d
	ENDDO
	IF (deltaT_0 .EQ. 0.0) GOTO 5250

* for monthly data delta-T may vary from 28 to 31, so allow a margin of error
	epsilon = 1.0001 * (31.-28.)/28.

* now re-evaluate the deltas to find the lowest common denominator
* i.e. if we discover there is a gap of size 1.5*deltaT, that means that
* the correct delta is new_delta= 0.5*deltaT.  This re-interprets our two
* gaps as 2*new_delta and 3*new_delta
	deltaFactor = 1
	deltaT = deltaT_0
	alteredDelta = .TRUE.
	DO WHILE (alteredDelta)
	   alteredDelta = .FALSE.
	   DO iset = 2, nsets
	      d = start_time(iset)-start_time(iset-1)
	      d_ratio = d/deltaT
* delta ratio must be an integer within epsilon threshold
	      d_ratio_int = FLOAT(INT(d_ratio + 0.49))
	      IF (ABS(d_ratio-d_ratio_int)/d_ratio .GT. epsilon) THEN
* ... we have the wrong deltaT -- try an integer fraction of it
	         deltaFactor = deltaFactor + 1
	         IF (deltaFactor .GT. 4) GOTO 5200
	         deltaT = deltaT_0 / deltaFactor
	         alteredDelta = .TRUE.
	         EXIT    ! try again with smaller delta
	      ENDIF
	   ENDDO	
	ENDDO

* loop through the data set list inserting dummys where there are gaps
	ngap = 0    ! not needed?
	nadded = 0
	iset = 1
	DO WHILE (iset .LT. nsets_out)
	   iset = iset + 1
	   d = start_time(iset)-start_time(iset-1)
	   IF (ABS(d-deltaT)/deltaT .GT. epsilon) THEN
* ... we have a gap (or at least an irregularity)
	      gap = d/deltaT
	      ngap = INT(gap + 0.5)      ! here ngap=1 means nothing missing
	      ngap = ngap - 1		 ! now  ngap=0 means nothing missing
	      nsets_out = nsets_out + ngap
	      IF (nsets_out .GT. maxmembers) THEN
	         nsets_out = maxmembers   ! needed to clean up files after err
	         GOTO 5200
	      ENDIF
*     ... open the gap
	      DO i = nsets_out, iset+ngap, -1
	         member_sets(i) = member_sets(i-ngap)
	         start_time (i) = start_time (i-ngap)
	      ENDDO
*     ... fill the gap
	      DO i = iset, iset+ngap-1
	         member_sets(i) = pdset_dummy
	      ENDDO
	      iset = iset + ngap
	   ENDIF
	ENDDO

	RETURN

* error exits
 5000	RETURN
 5100	CALL WARN(
     .		'Unable to sort forecast datasets due to corrupted times.')
	GOTO 5000
 5200	CALL ERRMSG( ferr_aggregate_error, status,
     .		'forecast start times are irregular or too gappy', *5000 )
 5250	CALL ERRMSG( ferr_aggregate_error, status,
     .		'some files have duplicate forecast start times', *5000 )

	END
