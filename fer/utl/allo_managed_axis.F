	INTEGER*4 FUNCTION ALLO_MANAGED_AXIS ( islot )

* Locates an unused axis (line) slot and returns it.
* This routine parallels TM_FIND_LINE_SLOT in function but whereas that
* routine allocates from lower memory in grid common and leaves gaps
* as axes are deleted this routine allocates from upper memory and uses
* linked lists to maintain integrity.
* If no axis slot is available this returns returns error.

* This code is a temporary version that does NOT use linked lists.  Instead
* it simply locates the first available axis slot starting from position one.
* By contrast TM_FIND_LINE_SLOT locates the beginning of the upper contiguous
* grid storage space

* This routine functions as a part of the TMAP library

* for FERRET version 4.00 8/94  *sh*
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* Arguments
	INTEGER*4	islot
*
* Parameters and commons
#ifdef unix
	include	'tmap_format/tmap_dims.parm'
#	include	"tmap_format/tmap_dset.parm"
	include	'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* Local declarations
	CHARACTER*13    TM_STRING
	INTEGER status

	DO 100 islot = 1, max_lines
	  IF (line_name(islot) .EQ. char_init16) THEN
	    ALLO_MANAGED_AXIS = merr_ok
	    RETURN
	  ENDIF
  100	CONTINUE

*
* No slot available
 9000	CALL TM_ERRMSG (merr_linelim, status,
     .	                'ALLO_MANAGED_AXIS', no_descfile, no_stepfile,
     .	                'MAX='//TM_STRING(DBLE(max_lines)), 
     .                   no_errstring,*9999)


 9999	ALLO_MANAGED_AXIS = status
	RETURN
	END
