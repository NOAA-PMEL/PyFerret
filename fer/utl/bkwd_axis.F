	LOGICAL FUNCTION BKWD_AXIS( idim, grid )

* determine whether a particular axis of a grid is backwards relative to
* the normal sense of a right hand coordinate system
* note: only backwards Z axes are supported

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/25/88

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER		idim, grid

* internal variable declarations:
	INTEGER		line
	CHARACTER*2	line_code

* check for program bug	! temp
	IF ( grid .EQ. unspecified_int4 ) STOP 'no_grd_orient'

* initialize
	line = grid_line( idim, grid )
	IF ( line .EQ. mnormal .OR. line .EQ. munknown ) THEN
	   BKWD_AXIS = .FALSE.
	   RETURN
	ELSE
	   line_code = line_direction( line )
	ENDIF

* see if it is an upside-down Z axis
	IF ( line_code .EQ. 'UD' ) THEN
	      BKWD_AXIS	   = .TRUE.
	ELSE
	      BKWD_AXIS	   = .FALSE.
	ENDIF

	RETURN
	END
