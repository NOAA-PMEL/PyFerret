    	SUBROUTINE SHOW_DYN_GRIDS(mods_cx)

* summarize the dynamic grids and axes currently in existence
* also do sanity checks that use counts are appropriate

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420:  11/95
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xtm_grid.cmn_text'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'          ! 12/94 SPLIT_LIST BUFFER
#endif

* calling argument declarations:
	INTEGER mods_cx

* local variable declarations:
	INTEGER grid, line, dset, uvar, count, i
	CHARACTER sanity_msg*24

* first show the dynamic grids
	grid = grid_flink(max_grids)
 	IF ( grid .EQ. max_grids ) THEN
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .						'No dynamic grids',16)
	ELSE
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .						'Dynamic grids:',14)
	ENDIF
* ... --- "while" loop
 100	IF ( grid .NE. max_grids ) THEN
* ... sanity check the use count -- must be in use by user vars
	   count = 0
	   DO 125 uvar = 1, max_uvar	
	      DO 120 dset = 0, max_gfdl_dsets
	         IF ( uvar_grid(uvar,dset) .EQ. grid ) count = count + 1
 120	      CONTINUE
 125	   CONTINUE
	   IF ( count .EQ. grid_use_cnt(grid) ) THEN
	      sanity_msg = ' '
	   ELSEIF ( count .GT. grid_use_cnt(grid) ) THEN
	      sanity_msg = 'err: grid under-allocated'
	   ELSE
	      sanity_msg = 'err: grid over-allocated'
	   ENDIF
* ... end of sanity check
	   WRITE (risc_buff,3000) grid_name(grid), grid_use_cnt(grid),
     .					sanity_msg
 3000	   FORMAT ( 4X,'GRID ',A, 'use count:',I4,5x,A)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   CALL SHOW_GRID(grid, mods_cx)
	   grid = grid_flink(grid)
	ENDIF
	IF (grid .NE.max_grids) GOTO 100

* now the dynamic axes
	line = line_flink(max_lines)
 	IF ( line .EQ. max_lines ) THEN
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .						'No dynamic axes',15)
	ELSE
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .						'Dynamic axes:',13)
	ENDIF
* ... --- "while" loop
 200	IF ( line .NE. max_lines ) THEN
* ... sanity check the use count -- must be in use by user vars
	   count = 0
* ... in use by static grids
	   DO 225 grid = 1, max_grids-1
	      DO 220 i = 1, 4
	         IF ( grid_line(i,grid) .EQ. line ) count = count + 1
 220	      CONTINUE
 225	   CONTINUE
* ... in use by dynamic grids
	   grid = grid_flink(max_grids)
 230	   IF ( grid .NE. max_grids ) THEN
	      DO 232 i = 1, 4
	         IF ( grid_line(i,grid) .EQ. line ) count = count + 1
 232	      CONTINUE
	      grid = grid_flink(grid)
	   ENDIF
	   IF (grid .NE.max_grids) GOTO 230
* ... in use by other axes
	   DO 235 i = max_lines, max_dyn_lines
	         IF ( line_parent(i) .EQ. line ) count = count + 1
 235	   CONTINUE
	   IF ( count .EQ. line_use_cnt(line) ) THEN
	      sanity_msg = ' '
	   ELSEIF ( count .GT. line_use_cnt(line) ) THEN
	      sanity_msg = 'err: axis under-allocated'
	   ELSE
	      sanity_msg = 'err: axis over-allocated'
	   ENDIF
* ... end of sanity check
	   WRITE (risc_buff,3010) line_name(line), line_use_cnt(line),
     .					sanity_msg
 3010	   FORMAT ( 4X,'AXIS ',A, 'use count:',I4,5X,A)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   line = line_flink(line)
	ENDIF
	IF (line .NE.max_lines) GOTO 200

	RETURN
	END

