	SUBROUTINE POS_LAB( ss, cx, idim, ndec, string, slen )

* create a string of the form (e.g.) "I:22  X:130.5E"
* if grid is unknown then "I:22", alone
* if subscript is unknown then "I:*"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   ???? 
*	 7/10/90 - added checks for invalid grid and ss

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER		ss, cx, idim, ndec, slen
	CHARACTER*(*)	string
	
* local variable declarations:
	INTEGER		TM_LENSTR1, grid
	CHARACTER*4	TM_FMT
	REAL*8		TM_WORLD

* initialize string with subscript (e.g.) "I:22 " or "I:*"
	IF ( ss .NE. unspecified_int4 ) THEN
	   string = ss_dim_name(idim) // ':' // TM_FMT( FLOAT(ss),1,4,slen )
	   slen = slen + 2
	ELSE
	   string = ss_dim_name(idim) // ':*'
	   slen = 3
	   RETURN
	ENDIF

* done with pathological cases
	IF ( cx .GT. 0 ) THEN
	   grid = cx_grid( cx )
	ELSE
	   RETURN
	ENDIF

* add on world coord axis name (e.g.) "I:22  X:"
	string = string(:slen+1) // ww_dim_name(idim) // ':'

* add formatted world coordinate position
	CALL TRANSLATE_TO_WORLD
     .	   ( TM_WORLD( ss, grid, idim, box_middle ),
     .	     idim, grid, ndec, string(slen+7: ) )

* compute string length
	slen = TM_LENSTR1( string )

	RETURN
	END
