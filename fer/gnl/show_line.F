	SUBROUTINE SHOW_line( line, brief )

* display basic facts about the given line

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/1/88
* V200:  9/22/89 - added "orient" for axes without particular orientations
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xunits.cmn_text'
	external xunits_data
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	LOGICAL		brief
	INTEGER 	line

* local variable declarations:
	INTEGER		idim, orient
	CHARACTER	t0_head*2, t0_col*20

* determine orientation
	DO 100 idim = 1, 5
	   IF ( line_direction(line) .EQ. axis_orients( idim ) ) THEN
	      orient = idim
	      GOTO 110
	   ENDIF
 100	CONTINUE
	orient = 0
	idim   = 1			! use X slot for non-oriented
 110	IF ( idim .EQ. 5 ) idim = 3	! DU --> UD on Z axis

* create a dummy grid with this axis
* (kludge to use existing code that works with grids)
	grid_line(idim, mgrid_buff) = line

* report basic facts using code from SHOW_GRID
	CALL LINE_FACTS( ttout_lun, idim, mgrid_buff, orient )
	IF ( brief ) RETURN

* report /T0= for formatted time axis
	IF ( line_direction(line) .EQ. 'TI' ) CALL SPLIT_LIST
     .		(pttmode_explct, ttout_lun, 'T0 = '//line_t0(line), 0)

	RETURN
	END
