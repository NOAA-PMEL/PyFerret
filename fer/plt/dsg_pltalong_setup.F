	SUBROUTINE DSG_PLTALONG_SETUP (dset, plot_orient, its_traj, 
     .				   dsg_as_traj, dsg_as_time, status)

* Get PLOT/ALONG= and return the orientation to use in plotting for a DSG dataset
*
* TrajectoryProfile	PLOT/ALONG=xy  ==> trajectory	(dsg_as_traj is set)
*			PLOT/ALONG=z   ==> prof
* TimeseriesProfile	PLOT/ALONG=xy  ==> point
*			PLOT/ALONG=z   ==> prof
*			PLOT/ALONG=t   ==> time		(dsg_as_time is set)
* Profile		PLOT/ALONG=xy  ==> point
* Timeseries		PLOT/ALONG=xy  ==> point
* Trajectory		PLOT/ALONG=t   ==> time
*
* as well, all of the types can have along= to their native
* direction, e.g. Timeseries can have /along=t, etc
* 

        IMPLICIT NONE
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'

#include "tmap_dset.parm"

* arguments9/9/2020
	INTEGER dset, plot_orient, status
	LOGICAL its_traj, dsg_as_traj, dsg_as_time

* Local variables
	INTEGER orientation, loc, along_dim
	CHARACTER buff1*1, message*128

* Initialize
	message = 'none'
	status = ferr_ok

	dsg_as_time = .FALSE.
	dsg_as_traj = .FALSE.

	IF (dset .LE. pdset_irrelevant) RETURN

	its_traj = dsg_feature_type(dset).EQ.pfeatureType_Trajectory .OR.  
     .	          dsg_feature_type(dset).EQ.pfeatureType_Point

* dataset orientation. 

	orientation = dsg_orientation(dset) 

* Is there a PLOT/ALONG= qualifier?

	along_dim = no_dim
	loc = qual_given(slash_plot_along) 
	IF ( loc .GT. 0. )  THEN
	      
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                     buff1, status )
	   IF (status .NE. ferr_ok) THEN
	     plot_orient = orientation
	     GOTO 4000
	   ENDIF

	   DO along_dim = 1, nferdims
	     IF ( buff1 .EQ. ww_dim_name(along_dim) ) EXIT
	   ENDDO

	ENDIF

	IF (along_dim .EQ. no_dim) GOTO 4000

* Check valid setings for PLOT/ALONG= for each Feature Type.
* When the setting agrees with the default plot direction, ignore
* the /along= setting.

	IF (orientation .EQ. pfeatureType_Point) THEN
	
	   plot_orient = orientation
	   IF (along_dim .EQ. x_dim) THEN
	      GOTO 5000
	   ELSE
	      message = 'PLOT/ALONG='//ww_dim_name(along_dim)//
     .	      ' : Point data can only be plotted along XY'
	   ENDIF
	ELSE IF (orientation .EQ. pfeatureType_Trajectory) THEN

	   IF (along_dim .EQ. x_dim) THEN
	      GOTO 5000	      
	   ELSE IF (along_dim.EQ.t_dim) THEN
	      plot_orient = pfeatureType_TimeSeries
	   ELSE
	      message = 'PLOT/ALONG='//ww_dim_name(along_dim)//
     .	      ' : Trajectory data can only be plotted along XY or T'
	   ENDIF


	ELSE IF (orientation .EQ. pfeatureType_Profile) THEN

	   IF (along_dim .EQ. z_dim) THEN
	      GOTO 5000	      
	   ELSE IF (along_dim .EQ. x_dim) THEN
	      plot_orient = pfeatureType_Point
	      dsg_as_traj = .TRUE.
	   ELSE
	      message = 'PLOT/ALONG='//ww_dim_name(along_dim)//
     .	      ' : Profile data can only be plotted along XY or Z'
	   ENDIF

	ELSE IF (orientation .EQ. pfeatureType_TimeSeries) THEN

	   IF (along_dim .EQ. t_dim) THEN
	      GOTO 5000	      
	   ELSE IF (along_dim .EQ. x_dim) THEN
	      plot_orient = pfeatureType_Point
	      dsg_as_traj = .TRUE.
	   ELSE
	      message = 'PLOT/ALONG='//ww_dim_name(along_dim)//
     .	      ' : Timeseries data can only be plotted along XY or Z'
	   ENDIF

	ELSE IF (orientation .EQ. pfeatureType_TrajectoryProfile ) THEN

	   IF (along_dim .EQ. x_dim) THEN 
	      plot_orient = pFeatureType_Trajectory
	      dsg_as_traj = .TRUE.
	   ELSE IF (along_dim .EQ. z_dim) THEN
	      GOTO 5000
	   ELSE
	      message = 'PLOT/ALONG='//ww_dim_name(along_dim)//
     .	      ' : TrajectoryProfile data'//
     .	      '  can only be plotted along XY or Z'
	   ENDIF
	   
	ELSE IF (orientation .EQ. pfeatureType_TimeseriesProfile ) THEN

	   IF (along_dim .EQ. x_dim) THEN 
	      plot_orient = pFeatureType_Point
	      dsg_as_traj = .TRUE.
	   ELSE IF (along_dim .EQ. z_dim) THEN
	      GOTO 5000
	   ELSE IF (along_dim .EQ. t_dim) THEN
	      plot_orient = pfeatureType_Timeseries
	      dsg_as_time = .TRUE.
	   ELSE
	      message = 'PLOT/ALONG='//ww_dim_name(along_dim)//
     .	      ' : pfeatureType_TimeseriesProfile data'//
     .	      ' can only be plotted along XY, Z, or T'
	   ENDIF

	ENDIF

 4000	CONTINUE

	its_traj = plot_orient.EQ.pfeatureType_Trajectory .OR.  
     .	          plot_orient.EQ.pfeatureType_Point .OR.  
     .	          dsg_as_traj

	IF (message .NE. 'none') THEN
	   CALL ERRMSG (ferr_invalid_command,  status, message, *5000)
	ENDIF

 5000	RETURN
	END
