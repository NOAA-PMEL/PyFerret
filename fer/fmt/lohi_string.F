	CHARACTER*(*) FUNCTION LOHI_STRING( lo,hi,idim,grid,
     .                                      to_word,ndec,slen )

* create a string describing a world coordinate range
* for example: 160E:140W or 10m (if lo depth = hi)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  1/10/89 - extracted from LOHI_WORLD 0.3
* V200:  2/27/90 - allowed unspecified limits as "???"
* V230:  2/10/92 - SUN port: must use a temporary string, strbuf, instead of
*                  routine name LOHI_STRING internally.

#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations
	INTEGER	idim, grid, ndec, slen
	CHARACTER*(*) to_word
	REAL*8	lo, hi

* internal variable declarations:
	INTEGER	sub_len, full_len
        CHARACTER*80 strbuf       ! needed for SUN FORTRAN

* initialize
	full_len = LEN( LOHI_STRING )
	strbuf = ' '

* always put in lower limit
	IF ( lo .NE. unspecified_val8 ) THEN
	   CALL TRANSLATE_TO_WORLD( lo, idim, grid, ndec, strbuf )
	ELSE
	   strbuf = '???'
	ENDIF
	CALL LEFT_JUST( strbuf, strbuf, slen )

* put in upper limit if different from lower
	IF ( hi .NE. lo ) THEN
	   strbuf = strbuf(:slen) // to_word	! either " to " or ":"
	   slen = MIN( full_len, slen + LEN(to_word) + 1 )
	   CALL TRANSLATE_TO_WORLD( hi, idim, grid, ndec, strbuf(slen:) )
	   CALL LEFT_JUST( strbuf(slen:), strbuf(slen:), sub_len )
           slen = MIN( full_len, slen + sub_len - 1 )
	ENDIF
	
        LOHI_STRING = strbuf

	RETURN
	END
