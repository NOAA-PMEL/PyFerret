	BLOCK DATA XALT_MESSAGES_DATA

* initialize data for indicated COMMON block

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* 2/28/95

#ifdef unix
	include 'tmap_format/xalt_messages.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XALT_MESSAGES.CMN_TEXT'
#endif

	DATA	alt_messages / .FALSE. /

	DATA	alt_nlines / 0 /

	END
