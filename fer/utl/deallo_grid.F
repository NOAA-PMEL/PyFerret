	SUBROUTINE DEALLO_GRID( status )

* deallocate a scratch space from the grid COMMON stack

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/22/89
* V420: 10/95 - deallocate any dynamic lines that were used by this grid
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	INTEGER	status

* internal declarations:
	INTEGER	idim

* clean out the grid deallocating any dynamic lines that this used
	DO 100 idim = 1, 4
	   IF (grid_line(idim,grd_stk_ptr) .GT. max_lines) CALL
     .			TM_DEALLO_DYN_LINE(grid_line(idim,grd_stk_ptr) )
	   grid_line(idim,grd_stk_ptr) = unspecified_int4
 100	CONTINUE
	grid_name(grd_stk_ptr) = char_init16

* remove the stack slot
	grd_stk_ptr = grd_stk_ptr + 1

	IF ( grd_stk_ptr .GT. max_grids ) CALL ERRMSG
     .		( ferr_stack_ovfl, status, 'deallo_grid',*5000 )

	status = ferr_ok

 5000	RETURN
	END
