	LOGICAL FUNCTION TM_HAS_STRING( string, substring )

* check in a case-independent search to see if "substring" is in "string"
* note: string and substring can be at most 80 characters

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 02/11/88
* revision 0.01 - 05/03/88 - made temp strings CHAR*255 (this is the max length
*			     that will work correctly (MAV)
* revision 0.10 - 01/20/91 - replaced VAX runtim library reference with INDEX
*                            FORTRAN intrinsic function

* calling argument declarations
	CHARACTER*(*)	string, substring

* local variable declarations
	INTEGER	STR_UPCASE, TM_LENSTR1,
     .		vms_stat, sublen, strlen
	CHARACTER*255	up_string, up_sub

* length of substring
	sublen = TM_LENSTR1( substring )
	strlen = TM_LENSTR1( string )
	IF (sublen .GT. 255 .OR. strlen .GT. 255)
     .	  STOP 'STRING TRUNCATION IN TM_HAS_STRING - CONTACT PROGRAMMER'

* convert to upper case
	vms_stat = STR_UPCASE( up_string, string )
	vms_stat = STR_UPCASE( up_sub, substring(:sublen) )

* find the substring
	TM_HAS_STRING = INDEX( up_string, up_sub(:sublen) ) .GT. 0

	RETURN
	END
