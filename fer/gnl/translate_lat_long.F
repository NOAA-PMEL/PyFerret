	SUBROUTINE TRANSLATE_LAT_LONG(string, idim, answer, formatted, status)

* convert an ascii representation of a latitude or longitude value into 
* a floating point value using special format rules

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/11/86
* revision 0.1 - 3/14/89 - return "formatted" flag
* V300  9/93 - insist that "N", "S", "E", "W" are the LAST character
* V420 (IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"

* possible formats:
*   x coord:  #  or #E  or  #W  for East  or West,  respectively
*   y coord:  #  or #S  or  #N  for South or North, respectively
*   z coord:  #
*   t coord:  #  or dd-mmm-yyyy@hh-mm-ss  eg 14-JAN-1986@11:15

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

	CHARACTER*(*)	string
	LOGICAL		n_given, s_given, e_given, w_given, formatted
	INTEGER		n_pos, s_pos, e_pos, w_pos, slen, status, idim
	REAL*8		answer

* local declarations
	CHARACTER	buff*32		! 10/95

* initialize
	slen = LEN ( string )
	n_given = string(slen:slen).EQ.'N' .OR. string(slen:slen).EQ.'n'
	s_given = string(slen:slen).EQ.'S' .OR. string(slen:slen).EQ.'s'
	e_given = string(slen:slen).EQ.'E' .OR. string(slen:slen).EQ.'e'
	w_given = string(slen:slen).EQ.'W' .OR. string(slen:slen).EQ.'w'
	formatted = n_given .OR. s_given .OR. e_given .OR. w_given

* how much of the string is numeric ?
	IF ( formatted ) slen = slen - 1

* decode the value
	buff =  string(1:slen)		! 10/95
	READ ( buff, * , ERR=5000 ) answer

* make corrections for latitude/longitude notation
	IF     ( n_given ) THEN

* latN - no change
	   IF ( idim .NE. y_dim ) GOTO 5000
	ELSEIF ( s_given ) THEN

* latS - make negative
	   IF ( idim .NE. y_dim ) GOTO 5000
	   answer = -answer
	ELSEIF ( e_given ) THEN

* longE - no change
	   IF ( idim .NE. x_dim ) GOTO 5000
	ELSEIF ( w_given ) THEN

* longW - convert	
	   IF ( idim .NE. x_dim ) GOTO 5000
	   answer = 360. - answer
	ENDIF

	status = ferr_ok
	RETURN

* error exit
 5000	CALL ERRMSG	( ferr_syntax, status, string, *5010 )
 5010	RETURN
	END
