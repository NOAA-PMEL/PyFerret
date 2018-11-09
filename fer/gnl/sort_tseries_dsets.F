	SUBROUTINE SORT_TSERIES_DSETS (nmembers, nsteps, skip_edges,
     .				       memb_nsteps, memb_order, memb_sort)
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

* sort the time steps that have been pulled from the time series members
*
* V698 *sh* 1/16 

* calling arguments
*   nmembers	- number of members in aggregation (input)
*   nsteps	- total number of time steps in agg axis (input)
*   skip_edges	- flag to turn off cell edge processing (input)
*   memb_nsteps	- number of time steps from each member file (input)
*   memb_order	- ordering index of member files (input empty/output filled)
*   memb_sort   - starting time steps as currently ordered, to be sorted (in/out)

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
	LOGICAL skip_edges
	INTEGER nmembers, nsteps, memb_nsteps(nmembers), memb_order(nmembers)
	REAL    memb_sort(nmembers)

* local variable declarations:
	LOGICAL sorted
	INTEGER i
	REAL    rorder(nmembers)

* initialize
	sorted = .FALSE.

* ... pre-sorted indices in a REAL array
	DO i = 1, nmembers
	   rorder(i) = i
	ENDDO

* sort the datasets by start date
	CALL HEAP2(memb_sort, rorder, unspecified_val8, nmembers)

* Return the sort order and the total number of steps

	nsteps = 0.
	DO i = 1, nmembers
	   memb_order(i) = INT(rorder(i))
	   IF (memb_order(i) .NE. i) sorted = .TRUE.
	   nsteps = nsteps + memb_nsteps(i)
	ENDDO

	RETURN
	END
