	LOGICAL FUNCTION GEOG_COS_FACTOR( idim, grid )

* determine whether a particular axis of a grid represents geographical
* locations (lat.,long., etc.) requiring COS corrections

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/25/88
* V230:  7/23/92 - TRUE only if grid has a valid latitude axis (major rewrite)

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'tmap_format/xunits.cmn_text'
        external xunits_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER		idim, grid

* internal variable declarations:
	INTEGER		xaxis, yaxis

* check for program bug	! temp
	IF ( grid .EQ. unspecified_int4 ) STOP 'no_grd_orient'

* initialize
	xaxis = grid_line( x_dim, grid )
	yaxis = grid_line( y_dim, grid )

* check that the X and Y axes of this grid are both degrees
* assume the calling routine calls this only for lat or long
*     (so the orientation of idim isn't checked here)

	IF ( xaxis .EQ. mnormal .OR. xaxis .EQ. munknown
     .  .OR. yaxis .EQ. mnormal .OR. yaxis .EQ. munknown ) THEN
	   GEOG_COS_FACTOR = .FALSE.
	ELSE
           GEOG_COS_FACTOR = line_unit_code(xaxis) .EQ. pun_degrees
     .                 .AND. line_unit_code(yaxis) .EQ. pun_degrees
	ENDIF

	RETURN
	END
