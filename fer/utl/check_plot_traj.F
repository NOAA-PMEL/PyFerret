	SUBROUTINE CHECK_PLOT_TRAJ (versus, nvars, grid, its_traj)

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
* Are we making a plot from a DSG trajectory dataset?
* Is it a trajectory plot? 
* Is it a non-trajectory ribbon plot from a traj dataset?
* Is this just a regular ribon plot with 3 variables (not a lon/lat traj plot)? 
* OR a PLOT/VS var1,var2 from a DSG trajectory dataset for var1,var2 not lon/lat?


* V751 7/19 *acm*  Changes for working with user-defined variables on DSG grid
* V76  1/20 *acm*  Point data are also plotted as ribbon plots, with symbols.

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
        include 'xdset_info.cmn_text'
	include 'xplot_setup.cmn'
#include "tmap_dset.parm"

* calling argument declarations:
	LOGICAL versus, its_traj
	INTEGER nvars, grid

* internal variable declarations:
	INTEGER	TM_DSG_DSET_FROM_GRID, TM_UNIT_ID, dset, iunits1, iunits2, cx
	CHARACTER VAR_UNITS*40, buff1*40


	dset = TM_DSG_DSET_FROM_GRID( grid )
       its_traj = dsg_feature_type(dset).EQ.pfeatureType_Trajectory .OR.  
     .            dsg_feature_type(dset).EQ.pfeatureType_Point

	IF (its_traj .AND. versus .AND. nvars.GT.1) THEN

	   cx = is_cx(1)
	   buff1 = VAR_UNITS(cx)
	   iunits1 = TM_UNIT_ID(buff1)
	       
	   cx = is_cx(2)
	   buff1 = VAR_UNITS(cx)
	   iunits2 = TM_UNIT_ID(buff1)
	       
	   its_traj = (iunits1.EQ.4 .AND. iunits2.EQ.4)  ! still a traj plot
	   IF (.NOT.its_traj .AND. nvars.EQ.2) ribbon_plot = .FALSE.
	ENDIF

	RETURN
	END
