	INTEGER FUNCTION TM_GET_GRIDNUM (user_grid)
* Searches the list of predefined and user-defined grid_names and returns the
* number associated with the grid_name given by the user. If no match is found
* the value unspecified_int4 is returned.
*
* written 11/86 By M. Verschell for PMEL/TMAP
*
* revision 0.01 - 08/17/88 - changed name from GT_GET_GRIDNUM
*
* *kob* 10/96 	- Linux port.  Linux didn't like tabs between "include"
*	          and what was being included.  removed them

#ifdef unix
# include "tmap_format/tmap_dset.parm"	
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

	CHARACTER*(*)	user_grid
	INTEGER*4	ngrid, istat
	INTEGER*4	STR_CASE_BLIND_COMPARE

	DO 100 ngrid=1,max_grids
	  istat = STR_CASE_BLIND_COMPARE(grid_name(ngrid),user_grid)
	  IF (istat .EQ. str_match) THEN
	    TM_GET_GRIDNUM = ngrid
	    GOTO 900
	  ENDIF
100	CONTINUE

	TM_GET_GRIDNUM = unspecified_int4

900	RETURN
	END
