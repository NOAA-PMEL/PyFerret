	SUBROUTINE TM_NOTE ( string, lunit )

* display a warning message to the operator

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 2/92 for TMAP library (from FERRET rouitine "WARN")
* *sh* 2/28/95 - allow message to be sent to alternative location

* calling argument declarations:
        INTEGER lunit
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		TM_LENSTR1, slen

#ifdef unix
	include 'tmap_format/xalt_messages.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XALT_MESSAGES.CMN'
#endif

* string length
	slen = TM_LENSTR1( string )

* do it
	IF ( alt_messages ) THEN
	   CALL TM_ALTERNATE_NOTE( string, lunit )
	ELSE
	   WRITE ( lunit, 3000 ) string(:slen)
 3000	   FORMAT ( ' *** NOTE: ',A )
	ENDIF

	RETURN
	END
