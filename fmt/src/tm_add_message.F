	SUBROUTINE TM_ADD_MESSAGE( next_line )

* append a line of text to the alternate message buffer

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* 2/28/95

#ifdef unix
	include 'tmap_format/xalt_messages.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XALT_MESSAGES.CMN_TEXT'
#endif

* calling arguments
	CHARACTER*(*) next_line

	IF ( alt_nlines.LT.max_alt_text ) THEN
	      alt_nlines = alt_nlines + 1
	      alt_text(alt_nlines) = next_line
	ENDIF

	RETURN
	END
