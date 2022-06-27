       SUBROUTINE INIT_T_AGG_MC_DSET(dset_num, nagfiles, memb_filename,
     .            memb_tstep_start, memb_tstep_end, memb_nstep_start, 
     .            memb_nsteps, memb_order, status)

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
*
* Initialize a multifile netCDF ("MC") T-aggregated dataset
*    - information equivalent to reading stepfile information from descriptor
*      file is passed in through the argument list
*    - call cd_init_dset on first file of data set to set up all 
*      common information for the dataset, such as variables, etc
*    - assemble the T-aggregate dataset and associate this
*      time axis with the parent grids
*
* V695+ *sh* 10/15 - new, hack-built from MC_INIT_DSET
* V698 *sh*   2/16 - member files may be disordered
* v767 6/22 *acm* See issue 112: Detect whether T dimension becomes too 
*                 large for 4-byte integer indexing



* arguments:
*	dset_num	 - number assigned to this data set (input)
*	nagfiles	 - number of files in the aggregation (input)
*	memb_filename	 - array of filenames (input)
*	memb_tstep_start - array of first timestep in each file (input)
*	memb_tstep_end	 - array of last  timestep in each file (input)
*       memb_nstep_start - array of time axis indices of the first step (input)
*	memb_nsteps	 - array of N timesteps in each file (input)
*	memb_order	 - sequence ordering of member (input)
*	status		 - return status (output)

        IMPLICIT NONE
* Argument definitions
        INTEGER dset_num, nagfiles, memb_order(nagfiles), 
     .     memb_nstep_start(nagfiles), memb_nsteps(nagfiles), status
        REAL*8 memb_tstep_start(nagfiles), memb_tstep_end(nagfiles)
        CHARACTER*512	memb_filename(nagfiles)

* Parameter and common files

	include 'tmap_errors.parm'		! error codes
#include "tmap_dset.parm"	
	include 'tmap_dims.parm'		! data set dimensions
!#include "gt_lib.parm"
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xio.cmn_text'
        external xio_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xdset_info.cmn_text'		! data set common
	external xdset_info_data
	include 'xstep_files.cmn_text'		! step file common
	external xstep_files_data
	include 'xdset_info.cd_equiv'		! has mc_init_stepfile

* Define variables from Set File namelist records

	include 'descript.def'


* Internal declarations

	CHARACTER*13	TM_STRING
	INTEGER		ivar, stp_pos,
     .			istep, i, nrecperstep, epos, first_file,
     .			stp_cnt, min_pos, max_pos,istat,
     .			lline, file_reclen
	REAL*8		file_trange, steps_total

* simulate reading of step records - one per file to be aggregated
	istep = 0

* Initialize some STEPFILE variables
	s_aux_set_num = 0       ! irrelevant?
	s_regvarflag  = ' '     ! irrelevant
	s_index       = ' '	! irrelevant
	file_reclen = unspecified_int4  ! irrelevant?
	s_delta	      = 0	! used to determine if regular time steps

* * * * * * top of stepfile loop * * * * *

	steps_total = 0
	DO i = 1, nagfiles
	  istep = memb_order(i)

* calculate time range per file and check for even number per file
	  file_trange = memb_tstep_end(istep)-memb_tstep_start(istep)
	  s_delta = file_trange / DBLE(memb_nsteps(istep))

* Find the next storage position for stepfiles in common
	  IF (i .EQ. 1) THEN
* ... stepfile of first file in member list was assigned during netCDF init
	     DO stp_cnt = 1,maxstepfiles
	       IF (sf_setnum(stp_cnt) .EQ. dset_num) GOTO 100
	     ENDDO
	     GOTO 5100     ! should never happen
	  ELSE  
	     DO stp_cnt = stp_pos,maxstepfiles
	       IF (sf_setnum(stp_cnt) .EQ. set_not_open) GOTO 100
	     ENDDO
* No room left for new stepfiles
	     CALL TM_ERRMSG ( merr_filim, status, 'MC_AGG_INIT',
     .			   dset_num, no_stepfile,
     .			   'MAX='//TM_STRING(DBLE(maxstepfiles)),
     .			   no_errstring, *9900)
	   ENDIF

* store values in common
 100	  sf_setnum(stp_cnt) = dset_num
	  sf_name  (stp_cnt) = memb_filename(istep)
	  sf_1step (stp_cnt) = memb_tstep_start(istep)
	  sf_lastep(stp_cnt) = memb_tstep_end(istep)
* sf_ndxs and sf_ndxe contain the L limits for each file on the agg T axis
	  sf_ndxs  (stp_cnt) = memb_nstep_start(istep)
	  sf_ndxe  (stp_cnt) = sf_ndxs(stp_cnt) + memb_nsteps(istep) - 1
	  sf_delta(stp_cnt) = s_delta
!	  sf_aux_set_num(stp_cnt) = s_aux_set_num
	  sf_lunit(stp_cnt) = file_not_open
	  sf_index(stp_cnt) = s_index
	  sf_reclen(stp_cnt) = file_reclen !???uninited var in MC_INIT_DSET
	  IF (s_regvarflag .EQ. ' ') sf_regvars(stp_cnt) = .TRUE.
!	  IF (last_delta .EQ. -999.) last_delta = s_delta  ! relevant?
	  IF (s_delta .NE. 0) THEN
	    sf_regtimes(stp_cnt) = .TRUE.		   ! relevant?
	  ELSE
	    sf_regtimes(stp_cnt) = .FALSE.
	  ENDIF

	  steps_total = steps_total + memb_nsteps(istep)
	  IF (steps_total .GT. arbitrary_large_int4) GOTO 5500

* time-independent fields are read from the first file in the member list
	  IF (i .EQ. 1) mc_init_stepfile(dset_num) = stp_cnt

* prepare for next file
	  stp_pos = stp_cnt+1

	ENDDO  
* * * * end of stepfile loop * * * * *


* summarize certain stepfile info at the dataset level
	ds_1step(dset_num)  = memb_tstep_start(memb_order(1))
	ds_lastep(dset_num) = memb_tstep_end(memb_order(nagfiles))
	IF (ds_regsteps(dset_num)) THEN
	  ds_delstep(dset_num)  = line_delta(ds_time_axis(dset_num))
	ELSE
	  ds_delstep(dset_num) = 0.
	ENDIF
	status = merr_ok

	RETURN


* errors
 9900	status = ferr_TMAP_error
 5000	RETURN
 5100	CALL ERRMSG( ferr_internal, status, 'init_t_agg_mc_dset', *5000)
 5500   CALL ERRMSG(ferr_grid_definition, status,
     .     'Aggregated dimension too long for 4-byte integer indexing: '
     .		//'Long dimensions not implemented', *5000)

	END
