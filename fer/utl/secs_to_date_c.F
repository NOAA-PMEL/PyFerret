	SUBROUTINE secs_to_date_c( bcsecs, hdate )

* C-callable routine to convert the number of seconds since BC to a date string

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V400: 11/94 

	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'

* calling argument declarations:
	REAL*8 bcsecs
#ifdef sun
        BYTE hdate(*)
#else
        INTEGER*1 hdate(*)
#endif

* local variable declarations
	INTEGER	  TM_LENSTR1, slen
	CHARACTER SECS_TO_DATE*20, temp_str*20

* do the conversion using FORTRAN CHARACTER variables
	temp_str = SECS_TO_DATE(bcsecs)

* now convert the result into a C string
	slen = TM_LENSTR1( temp_str )
	CALL TM_FTOC_STRNG( temp_str, hdate, slen )

	RETURN
	END

