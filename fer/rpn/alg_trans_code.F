	CHARACTER*3 FUNCTION ALG_TRANS_CODE( tcode )

* return the text associated with the given transformation code

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/4/87

* calling argument declarations:
	INTEGER	tcode

* internal variable declarations:
	INTEGER	i

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
#endif

* identify the code from known codes
	DO 100 i = 1, num_transforms
	   IF ( tcode .EQ. alg_trans_num( i ) ) GOTO 200
 100	CONTINUE

* unknown code
	ALG_TRANS_CODE = '???'
	RETURN

* return the text
 200	ALG_TRANS_CODE = alg_trans_text( i )
	RETURN
	END
