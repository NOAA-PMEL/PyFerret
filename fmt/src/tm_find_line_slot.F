	INTEGER*4 FUNCTION TM_FIND_LINE_SLOT ( islot )
*
* Locates an unused space in XGRID common for a new line. If no space available,
* returns error.
*
* revision 0.00 - 11/22/88
* revision 1.00 - 12/13/88 - changed to reverse order search
* bug fix - 10/2/91 - inserted comma in statement 9000 right before the *9999.  This error was
*                     caught by version 3 of f77, but not 2.1.    kob
* Arguments
	INTEGER*4	islot
*
* *kob* 10/96 	- Linux port.  Linux didn't like tabs between "include"
*	          and what was being included.  removed them
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
	DO 100 igrd = max_lines,1,-1
	  IF (line_name(igrd) .NE. char_init16) THEN
	    IF (igrd .EQ. max_lines) GOTO 9000
	    islot = igrd + 1
	    GOTO 200
	  ENDIF
  100	CONTINUE
	islot = 1
  200	TM_FIND_LINE_SLOT = merr_ok
	GOTO 9999
*
* No slot available
 9000	CALL TM_ERRMSG (merr_linelim, TM_FIND_LINE_SLOT,
     .	                'TM_FIND_LINE_SLOT', no_descfile, no_stepfile,
     .	                'MAX='//TM_STRING(DBLE(max_lines)), 
     .                   no_errstring,*9999)


 9999	RETURN
	END
