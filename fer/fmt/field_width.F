	INTEGER FUNCTION FIELD_WIDTH( val, grid, idim, ndec, numbers )

* determine the width of field needed to write out value "val" in a formatting
* style appropriate to axis "idim" of grid "grid" with ndec decimal points
* also return the number of characters needed exclusive of the extra formatting
* ("numbers")

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/18/88
* 10/2/95 - bug fix:  date string was using ndec instead of ABS(dec) for
*			determining field width

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	INTEGER	grid, idim, ndec, numbers
	REAL*8	val

* internal variable declarations:
	LOGICAL	GEOG_LABEL, fancy
	INTEGER	decim, left
	REAL	fval

* initialize
	fancy = GEOG_LABEL( idim, grid )	! special axis formatting
	decim = ABS( ndec )
	IF ( decim .GT. 0 ) decim = decim + 1	! allow room for decimal pt.
	fval = val

* get actual numerical value to format if different from given
	IF ( fancy ) THEN
	   IF ( idim .EQ. t_dim ) THEN
	      FIELD_WIDTH = date_str_len( MAX( 1, MIN(6,ABS(ndec)) ) )	! 10/95
	      RETURN
	   ELSEIF ( idim .EQ. x_dim .AND. fval .GT. 180.0 ) THEN
	      fval = 360. - val
	   ELSEIF ( idim .EQ. y_dim .AND. fval .LT. 0.0   ) THEN	
	      fval = -fval
	   ENDIF
	ENDIF

* digits to left of decimal
	IF ( ABS(fval) .LT. 10.0 ) THEN
	   left = 1
	ELSE
	   left = LOG10( ABS(fval) ) + 1
	ENDIF
	IF ( fval .LT. 0.0 ) left = left + 1	! minus sign

* numeric field width
	numbers = left + decim

* total field width
	IF ( fancy ) THEN
	   IF ( idim .EQ. x_dim .AND. fval .EQ. 180.0
     .	       .OR. idim .EQ. y_dim .AND. fval .EQ. 0.0	  ) THEN
	      FIELD_WIDTH = numbers
	   ELSE
	      FIELD_WIDTH = numbers + 1	! "E","N", etc. 
	   ENDIF

	ELSE
	   FIELD_WIDTH = numbers

	ENDIF

	RETURN
	END
