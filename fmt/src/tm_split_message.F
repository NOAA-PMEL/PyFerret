	SUBROUTINE TM_SPLIT_MESSAGE( chout, next_line )

* depending on the state of alt_messages either write this string to the
* given unit number or save it in a GUI buffer

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300:  6/16/93 - for prototype GUI interface

#ifdef unix
	include 'tmap_format/xalt_messages.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XALT_MESSAGES.CMN_TEXT'
#endif

* calling arguments
	INTEGER chout
	CHARACTER*(*) next_line

* internal variable declarations
	INTEGER TM_LENSTR1


	IF ( alt_messages ) THEN
	   CALL TM_ADD_MESSAGE( next_line )
	ELSE
	   WRITE ( chout, '(A)' ) next_line(:TM_LENSTR1(next_line))
	ENDIF

	RETURN
	END
