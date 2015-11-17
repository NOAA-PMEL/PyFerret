	SUBROUTINE DEFINE_T_AGGREGATION
     .		(dset_name, dset_title, nagfiles,
     .		 ag_filename,
     .		 gvn_ag_tstep_start, gvn_ag_tstep_end, gvn_ag_nstep,
     .		 t_regular, use_strict, dset_num, status)

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
* this routine functions as a wrapper around the routine DEFINE_T_AGG_MC_DSET
* it is intended to make that routine friendlier to call from Python
* Through this routine we permit the Python user to pass scalars to
* indicate the time step metadata is not available

* V695+ *sh* 10/15 new routine 

* arguments:
*   NOTE: the 3 args that begin with "gvn" are the "given" metadata for the
*   files to be aggregated.  If the gvn_ag_nstep array length is 1, then the
*   calling program has elected NOT to provide this information 

*	dset_name	- dataset name to assign to the aggregation (input)
*	dset_title	- dataset title (input)
*	nagfiles	- number of files in the aggregation (input)
*	ag_filename	- array of filenames (input)
*	gvn_ag_tstep_start - array of first timestep in each file (input)
*	gvn_ag_tstep_end   - array of last  timestep in each file (input)
*	gvn_ag_nstep	- array of N timesteps in each file (input)
*	t_regular	- flag to signal promise of regular timesteps (input)
*	use_strict	- flag to allow micro-adjustments of T axis (input)
*	dset_num	- number assigned to this data set (output)
*	status		- return status (output)
*

        include 'ferret.parm'
        include 'errmsg.parm'

* Argument definitions
	INTEGER		gvn_ag_nstep(*)
	REAL*8		gvn_ag_tstep_start(*), gvn_ag_tstep_end(*)
	LOGICAL		t_regular, use_strict
	INTEGER		nagfiles, dset_num, status
	CHARACTER*(*)	dset_name, dset_title
	CHARACTER*128	ag_filename(nagfiles)

* Internal declarations
	INTEGER	iset
	INTEGER	ag_nstep(nagfiles)
	REAL*8	ag_tstep_start(nagfiles), ag_tstep_end(nagfiles)

* ***!!! TEMPORARY  !!!*** -- need to expand error message list
	INTEGER ferr_invalid_pyFerret_call
	ferr_invalid_pyFerret_call = ferr_invalid_command

* initialize
	status = ferr_ok

* HOW TO GET THE SIZE OF AN ARRAY!!!!  Intrinsic SIZE function??


* has the caller of this routine provided the time step metadata?
	IF ( SIZE(gvn_ag_nstep).NE.SIZE(gvn_ag_tstep_start)
     .  .OR. SIZE(gvn_ag_nstep).NE.SIZE(gvn_ag_tstep_end)  ) GOTO 5100

	IF (SIZE(gvn_ag_nstep) .EQ. nagfiles) THEN
* yes -- copy the given metadata into working arrays
	   DO iset = 1, nagfiles
	      ag_nstep      (iset) = gvn_ag_nstep      (iset)
	      ag_tstep_start(iset) = gvn_ag_tstep_start(iset)
	      ag_tstep_end  (iset) = gvn_ag_tstep_end  (iset)
	   ENDDO
	ELSEIF (SIZE(gvn_ag_nstep) .EQ. 1) THEN
* no -- set up empty arrays
	   DO iset = 1, nagfiles
	      ag_nstep      (iset) = unspecified_int4
	      ag_tstep_start(iset) = unspecified_val8
	      ag_tstep_end  (iset) = unspecified_val8
	   ENDDO
	ELSE
* oops -- lengths of arrays are not conformable
	   GOTO 5100
	ENDIF

* now call the "real" routine
	CALL DEFINE_T_AGG_MC_DSET
     .		(dset_name, dset_title, nagfiles,
     .		 ag_filename,
     .		 ag_tstep_start, ag_tstep_end, ag_nstep,
     .		 use_strict, dset_num, status)


* success
	RETURN

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'aggregation timestep metadata  arrays must' //
     .		'contain metadata for the files or be all of length 1'
     .           *5000 )
	END
