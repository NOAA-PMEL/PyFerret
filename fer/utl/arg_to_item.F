	SUBROUTINE ARG_TO_ITEM

* transfer the list of arguments into a list of items

* this is a bit of a kludge recognizing that the Ferret command language
* is confusing if it requires uses to recognize the difference between
* "SHOW GRID A,B" and "SHOW GRID A B"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420:  4/96

* internal variable declarations:
	INTEGER	i

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* if the user did not separate items by commas then make it appear the he did

	IF ( num_args.GT.1 .AND. num_items.EQ.1 ) THEN
	   num_items = num_args
	   DO 100 i = 1, num_args
	      item_start(i) = arg_start(i)
	      item_end(i)   = arg_end(i)
 100	   CONTINUE
	ENDIF

	RETURN
	END	
