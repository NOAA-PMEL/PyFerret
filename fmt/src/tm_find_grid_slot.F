	INTEGER*4 FUNCTION TM_FIND_GRID_SLOT ( islot )
*
* Locates an unused space in XGRID common for a new grid. If no space available,
* returns error.
*
* revision 0.00 - 11/22/88
* revision 1.00 - 12/13/88 - changed to reverse order search 
* revision 1.01 - 07/13/89 - fixed missing comma in TM_ERROR call
* *kob* 10/96 	- Linux port.  Linux didnt like tabs 
*		  between "include"
*	          and what was being included.  removed them
*
* Arguments
	INTEGER*4	islot
*
* Parameters and commons
#ifdef unix
        include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_errors.parm'
        include 'tmap_format/xtm_grid.cmn_text'
        external xgt_grid_data
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif
*
* Local declarations
	CHARACTER*13	TM_STRING
	INTEGER*4	igrd
*
	DO 100 igrd = max_grids,1,-1
	  IF (grid_name(igrd) .NE. char_init16) THEN
	    IF (igrd .EQ. max_grids) GOTO 9000
	    islot = igrd + 1
	    GOTO 200
	  ENDIF
  100	CONTINUE
	islot = 1
  200	TM_FIND_GRID_SLOT = merr_ok
	GOTO 9999
*
* No slot available
 9000	CALL TM_ERRMSG (merr_gridlim, TM_FIND_GRID_SLOT,
     .	                'TM_FIND_GRID_SLOT', no_descfile, no_stepfile,
     .	                'MAX='//TM_STRING(DBLE(max_grids)),
     .			no_errstring, *9999)


 9999	RETURN
	END
