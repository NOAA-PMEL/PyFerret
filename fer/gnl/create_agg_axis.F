	SUBROUTINE CREATE_AGG_AXIS (len, line_dir, iline, status)

      IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	
* calling argument declarations:
	INTEGER	len, line_dir, iline, status

* local variable declarations:
	INTEGER ALLO_MANAGED_AXIS

* create the ensemble axis

	status = ALLO_MANAGED_AXIS (iline )
	IF ( status .NE. ferr_ok ) GOTO 5000

* save the line in memory - but don't catalog its name
	line_name     ( iline ) = 'ENSEMBLE'
	line_name_orig( iline ) = 'ENSEMBLE'
	line_start    ( iline ) = 1.0D0
	line_delta    ( iline ) = 1.0D0
	line_dim      ( iline ) = len
	line_regular  ( iline ) = .TRUE.
	line_modulo   ( iline ) = .FALSE.
	line_units    ( iline ) = ' '
	line_unit_code( iline ) = 0			! unknown units
	line_shift_origin( iline ) = .FALSE.

	line_direction( iline ) = 'NA'
	IF (line_dir .EQ. x_dim) THEN
	   line_direction( iline ) = 'WE'
	   line_name     ( iline ) = 'LONGUITUDE'
	ELSEIF (line_dir .EQ. y_dim) THEN
	   line_direction( iline ) = 'SN'
	   line_name     ( iline ) = 'LATITUDE'
	ELSEIF (line_dir .EQ. z_dim) THEN
	   line_direction( iline ) = 'UD'
	   line_name     ( iline ) = 'DEPTH'
	ELSEIF (line_dir .EQ. t_dim) THEN
	   line_direction( iline ) = 'TI'
	   line_name     ( iline ) = 'TIME'
	ELSEIF (line_dir .EQ. e_dim) THEN
	   line_direction( iline ) = 'EE'
	   line_name     ( iline ) = 'ENSEMBLE'
	ELSEIF (line_dir .EQ. f_dim) THEN
	   line_direction( iline ) = 'FI'
	   line_name     ( iline ) = 'FORECAST'
	ENDIF
	line_name_orig( iline ) = line_name ( iline )
	
 5000   RETURN	
 5300   status = ferr_TMAP_error
	RETURN

	END
