	SUBROUTINE CHECK_PLOT_TRAJ (versus, nvars, its_traj)

* Are we making a plot from a DSG trajectory dataset?
* Is it a trajectory plot? 
* Is it a non-trajectory ribbon plot from a traj dataset?
* Is this just a regular ribon plot with 3 variables (not a lon/lat traj plot)? 
* OR a PLOT/VS var1,var2 from a DSG trajectory dataset for var1,var2 not lon/lat?

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
        include 'xdset_info.cmn_text'
	include 'xplot_setup.cmn'
#include "tmap_dset.parm"

* calling argument declarations:
	LOGICAL versus, its_traj
	INTEGER nvars

* internal variable declarations:
	INTEGER	TM_UNIT_ID, iunits1, iunits2, cx
	CHARACTER VAR_UNITS*40, buff1*40


	cx = is_cx(1)
	its_traj = dsg_feature_type(cx_data_set(cx)) .EQ. pfeatureType_Trajectory

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
