	SUBROUTINE PURGE_MR_AXIS( old_axis, new_axis, status )

* eliminate all stored mrs that use a given axis
* and replace all references within grids to the old axis with the new axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  8/4/89 - from PURGE_USER_VAR
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include	'tmap_format/tmap_dims.parm'
#	include	"tmap_format/tmap_dset.parm"
	include	'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declaration:
	INTEGER	old_axis, new_axis, status

* internal variable declarations:
	INTEGER	grid, idim
	CHARACTER	I6*6

* the lowest axes are protected
	IF ( old_axis .LE. 7 ) GOTO 5200

* only regular axes may be re-defined
* (to avoid dealing with messy memory pointers)
!	IF ( .NOT. line_regular( old_axis ) ) GOTO 5100

* delete all variables in memory that are based on grids using this axis
	DO 100 grid = 1, max_grids
	  IF (grid_name(grid) .EQ. char_init16) GOTO 100

	  DO 50 idim = 1, 4
	     IF ( grid_line(idim,grid) .EQ. old_axis ) THEN
	         CALL PURGE_MR_GRID( grid, status ) 
	         IF ( status .NE. ferr_ok ) RETURN
	         GOTO 100
	     ENDIF
 50	  CONTINUE

 100	CONTINUE

* now we're safe from errors: replace the old axis with the new one
	DO 250 grid = 1, max_grids
	  IF (grid_name(grid) .EQ. char_init16) GOTO 250

	  DO 200 idim = 1, 4
	     IF ( grid_line(idim,grid) .EQ. old_axis ) THEN
	         grid_line(idim,grid) = new_axis
	     ENDIF
 200	  CONTINUE

 250	CONTINUE

* delete the old axis
	IF ( .NOT.line_regular(old_axis) ) THEN
	   WRITE ( I6,'(I6)' ) line_dim(old_axis)
	   CALL WARN( I6//' words of axis coordinate storage lost' )
	   line_subsc1( old_axis ) = unspecified_int4
	ENDIF
	line_name( old_axis ) = char_init16

	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
! 5100	CALL ERRMSG( ferr_grid_definition, status,
!     .		'cannot delete axes with irregular points: '
!     .		//line_name(old_axis),
!     .		*5000 )
 5200	CALL ERRMSG( ferr_grid_definition, status,
     .		'protected axis: '//line_name(old_axis), *5000 )
	END

