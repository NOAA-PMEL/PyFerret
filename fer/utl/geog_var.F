	LOGICAL FUNCTION GEOG_VAR( idim, cx )

* determine whether a particular axis of a VARIABLE represents geographical
* locations (lat.,long., etc.) requiring special label formatting

* V450: 3/97 - based on GEOG_LABEL

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
        include 'tmap_format/tmap_errors.parm'
	external xgt_grid_data
        include 'tmap_format/xunits.cmn_text'
        external xunits_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xtext_info.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		idim, cx

* internal variable declarations:
        LOGICAL         TM_DATE_OK
	INTEGER		TM_UNIT_ID, units
	CHARACTER*80	VAR_UNITS

* initialize
	GEOG_VAR = .FALSE.	! default

* see if labeling modes are disabled, first
	IF ( .NOT.ax_fmt(idim) )  RETURN

* get the "units" that goes with the variable (context)
	units = TM_UNIT_ID( VAR_UNITS(cx) )

* do these units imply a formatt-ible axis
	IF ( idim.EQ.t_dim .AND.
     .        (units.LT.pun_1st_time.OR.units.GT.pun_last_time)) THEN
	   GEOG_VAR = .FALSE.
!	   GEOG_VAR = .TRUE.		! where to get T0 ???
	ELSEIF ( (idim.EQ.x_dim.OR.idim.EQ.y_dim) .AND.
     .		units .EQ. pun_degrees ) THEN
	   GEOG_VAR = .TRUE.
	ELSEIF ( (idim.EQ.z_dim) .AND.
     .          ( units .EQ. pun_millibars
     .          .OR. units .EQ. pun_decibars
     .          .OR. units .EQ. pun_meters )  ) THEN
	   GEOG_VAR = .TRUE.
	ENDIF

	RETURN
	END
