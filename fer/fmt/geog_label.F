	LOGICAL FUNCTION GEOG_LABEL( idim, grid )

* determine whether a particular axis of a grid represents geographical
* locations (lat.,long., etc.) requiring special label formatting

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/25/88
* V230: 4/92 *sh*  only return .TRUE. for time axis if t0_date is valid
*       7/92 - use TM_DATE_OK for this
*            - vertical axis requires meters units AND UD orientation
* V320: 12/19/94 - recognize millibars and decibars as special axes

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
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER		idim, grid

* internal variable declarations:
        LOGICAL         TM_DATE_OK
	INTEGER		i, line
	CHARACTER*2	line_code

* check for program bug	! temp
	IF ( grid .EQ. unspecified_int4 ) STOP 'no_grd_orient'

* see if labeling modes are disabled, first
	IF ( .NOT.ax_fmt(idim) ) THEN
	   GEOG_LABEL = .FALSE.
           RETURN
	ENDIF

* initialize
	line = grid_line( idim, grid )
	IF ( line .EQ. mnormal .OR. line .EQ. munknown ) THEN
	   GEOG_LABEL = .FALSE.
	   RETURN
	ELSE
	   line_code = line_direction( line )
	ENDIF

* the use of line_direction is kludgy
* (it combines orientation, reversal, and formatting in an inadequate way)
        IF ( idim .LE. 2 ) THEN
           GEOG_LABEL = line_code .EQ. axis_orients(idim)
        ELSEIF ( idim .EQ. z_dim ) THEN
* for Z also insist on units of m because DU/UD is used to indicate reversal
           GEOG_LABEL =((line_code .EQ. 'UD'
     .            .AND. line_unit_code(line) .EQ. pun_meters)
     .    .OR.		(line_code .EQ. 'UD'
     .		   .AND. line_unit_code(line) .EQ. pun_millibars)
     .    .OR.		(line_code .EQ. 'UD'
     .		   .AND. line_unit_code(line) .EQ. pun_decibars) )
        ELSE
*  4/92 addition:  check for valid time axis T0
           GEOG_LABEL = line_code        .EQ. 'TI'
     .            .AND. TM_DATE_OK( line_t0(line) )
        ENDIF

	RETURN
	END
