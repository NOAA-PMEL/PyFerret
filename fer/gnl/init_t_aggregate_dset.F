	SUBROUTINE INIT_T_AGGREGATE_DSET(nagfiles, dset_name, dset_title,
     .			t_regular, use_strict, dset_num, status)
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
* assemble the needed arguments and call DEFINE_T_AGG_MC_DSET
* a close cousin of this routine is DEFINE_T_AGGREGATION, which similarly
* assembles the arguments, but is intended to for calling from Python
* Whereas Python has the option to supply the time axis metadata of each
* file to be aggregated, herein we will always need to glean that information
* by opening each file

* admittedly, the hierarchy of calling names is not self-evident.  They are:
* top level: INIT_T_AGGRETATE_DSET  and DEFINE_T_AGGREGATION (for Python)
*   ==> DEFINE_T_AGG_MC_DSET
*       ==> BUILD_T_AGG_AXIS
*       ==> INIT_T_AGG_MC_DSET

* V695+ *sh* 11/15 new routine 

* arguments:
*   NOTE: the 3 args that begin with "gvn" are the "given" metadata for the
*   files to be aggregated.  If the gvn_ag_nstep array length is 1, then the
*   calling program has elected NOT to provide this information 

*	dset_name	- dataset name to assign to the aggregation (input)
*	dset_title	- dataset title (input)
*	nagfiles	- number of files in the aggregation (input)
*	t_regular	- flag to signal promise of regular timesteps (input)
*	use_strict	- flag to allow micro-adjustments of T axis (input)
*	dset_num	- number assigned to this data set (output)
*	status		- return status (output)
*

* these are the arrays that need to be created
*	ag_filename	- array of filenames (input)
*	ag_tstep_start - array of first timestep in each file (input)
*	ag_tstep_end   - array of last  timestep in each file (input)
*	ag_nstep	- array of N timesteps in each file (input)


        include 'tmap_dims.parm'
	include 'ferret.parm'
        include 'errmsg.parm'
	include 'xprog_state.cmn'

* Argument definitions
	LOGICAL		t_regular, use_strict
	INTEGER		nagfiles, dset_num, status
	CHARACTER*(*)	dset_name, dset_title

* Internal declarations
	INTEGER	iset
	INTEGER	ag_nstep(nagfiles)
	REAL*8	ag_tstep_start(nagfiles), ag_tstep_end(nagfiles)
	CHARACTER*128	ag_filename(nagfiles)

* initialize
	status = ferr_ok

* set up array of filenames given by use on command line
	DO iset = 1, nagfiles
	  ag_filename(iset) = cmnd_buff(item_start(iset):item_end(iset))
	ENDDO

* set up dummy (empty) timestep metadata arrays
	DO iset = 1, nagfiles
	   ag_nstep      (iset) = unspecified_int4
	   ag_tstep_start(iset) = unspecified_val8
	   ag_tstep_end  (iset) = unspecified_val8
	ENDDO

* now call the "real" routine
	CALL DEFINE_T_AGG_MC_DSET
     .		(dset_name, dset_title, nagfiles,
     .		 ag_filename,
     .		 ag_tstep_start, ag_tstep_end, ag_nstep,
     .		 t_regular, use_strict, dset_num, status)


* done
	RETURN
	END
