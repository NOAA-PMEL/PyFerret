	SUBROUTINE TM_NEW_GRID_NAME ( suggested_name, new_name )
*
* If necessary creates a unique grid_name based on the suggested name by
* appending a number to the end of the suggested name

* Warning: if new_name is less than 5 characters this routine may STOP
*
* revision 0.00 - 11/22/88
*
* Arguments
	CHARACTER*(*)	suggested_name, new_name
*                                           
* Parameters and commons                    
#ifdef unix
        include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'  
	external xgt_grid_data
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'  
#endif
*
* Local declarations
	REAL		newnum
	INTEGER*4	suglen, outlen, addlen, uselen, igrd 
	INTEGER*4	TM_LENSTR1
	CHARACTER*4	TM_FMT, add_on
*
* initialize
	suglen   = TM_LENSTR1(suggested_name)
	outlen   = LEN( new_name )
	new_name = suggested_name
	newnum   = 0.

* check for an identical name
  10	DO 100 igrd = 1, max_grids
	  IF (grid_name(igrd) .EQ. new_name) GOTO 200
  100	CONTINUE

* clean bill of health - no name match
	RETURN
*
* yes there's a duplicate - try to "uniqueify" the name
  200	newnum = newnum + 1.
	add_on = TM_FMT(newnum, 4, 4, addlen )
	uselen  = MIN( suglen, outlen-addlen )
	IF ( uselen .LE. 0 ) STOP 'TM_NEW_GRID_NAME'
	new_name = suggested_name(:uselen)//add_on
*
* no danger of an infinite loop unless there are over 9999 names
	GOTO 10
	END
