	INTEGER FUNCTION TM_GET_LINENUM (user_line)
* Searches the list of pre-defined line_name's and returns the number
* associated with the line_name given by the user. If no match is found
* the value unspecified_int4 is returned.
*
* written 08/88 By M. Verschell for PMEL/TMAP
*
* revision 0.00 - 08/12/88
*
* *kob* 10/96 	- Linux port.  Linux didn't like tabs between "include"
*	          and what was being included.  removed them
* Include files
#ifdef unix
#include "tmap_format/tmap_dset.parm"	
	include 'tmap_format/tmap_dims.parm'	! 
	include 'tmap_format/gt_lib.parm'	! 
	include 'tmap_format/xtm_grid.cmn_text'	! pre-defined grid common
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'	! Data set constants
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'	! 
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'	! 
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'	! pre-defined grid common
#endif

	CHARACTER*(*)	user_line
	INTEGER*4	nline, istat
	INTEGER*4	STR_CASE_BLIND_COMPARE

	DO 100 nline=1,max_lines
	  istat = STR_CASE_BLIND_COMPARE(line_name(nline),user_line)
	  IF (istat .EQ. str_match) THEN
	    TM_GET_LINENUM = nline
	    GOTO 900
	  ENDIF
100	CONTINUE

	TM_GET_LINENUM = unspecified_int4

900	RETURN
	END
