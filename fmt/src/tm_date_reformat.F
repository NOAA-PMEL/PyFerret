	SUBROUTINE TM_DATE_REFORMAT( indate,  inform,
     .				     outdate, outform, status)
* This function will convert a date in format inform into a date in
* format outform

* *sh* 2/13/95
*
* Arguments:
*	indate	 - input date string
*	inform	 - code for input format
*	outdate	 - output date string
*	outform	 - code for output format
*	status   - returning status
*
	CHARACTER*(*)	indate, outdate
	INTEGER		inform, outform, status
*
* Inclusions
*
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

*
* Local definitions
*
	INTEGER	year, month, day, hour, minute, second, dlen

* Break up the input date string to it's various components
	CALL TM_BREAK_FMT_DATE ( inform, indate, year, month, day,
     .				    hour, minute, second, status )
	IF ( status .NE. merr_ok ) GOTO 9000

* and reassemble it into the desired output format
	CALL TM_MAKE_FMT_DATE ( outform, outdate, year, month, day,
     .				   hour, minute, second, dlen )

* Finished
	status = merr_ok
	RETURN

* Errors
 9000	RETURN
	END



