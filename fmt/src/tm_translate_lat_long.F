	SUBROUTINE TM_TRANSLATE_LAT_LONG (string,idim,answer,status)
*
* convert an ascii representation of a latitude or longitude value into 
* a floating point value using special format rules
*
* possible formats:
*   x coord:  #  or #E  or  #W  for East  or West,  respectively
*   y coord:  #  or #S  or  #N  for South or North, respectively
*   z coord:  #
*   t coord:  #  or dd-mmm-yyyy@hh-mm-ss  eg 14-JAN-1986@11:15
*
* programmer - Mark Verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 12/14/88 - from program by Steve Hankin
*
* include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/gt_lib.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

	CHARACTER*(*)	string
	LOGICAL		n_given, s_given, e_given, w_given
	INTEGER		n_pos, s_pos, e_pos, w_pos, end_pos, status, idim
	REAL*8		answer

* initialize
	n_pos = MAX ( INDEX(string,'N'), INDEX(string,'n') )
	s_pos = MAX ( INDEX(string,'S'), INDEX(string,'s') )
	e_pos = MAX ( INDEX(string,'E'), INDEX(string,'e') )
	w_pos = MAX ( INDEX(string,'W'), INDEX(string,'w') )

	n_given = n_pos .GT. 0
	s_given = s_pos .GT. 0
	e_given = e_pos .GT. 0
	w_given = w_pos .GT. 0

* how much of the string is numeric ?
	end_pos = LEN ( string )
	IF ( n_given.OR.s_given.OR.e_given.OR.w_given )	end_pos = end_pos - 1

* decode the value
	READ ( string(1:end_pos), * , ERR=9000 ) answer

* make corrections for latitude/longitude notation
	IF  ( n_given ) THEN

* latN - no change
	  IF ( idim .NE. y_dim ) GOTO 9000
	ELSEIF ( s_given ) THEN

* latS - make negative
	  IF ( idim .NE. y_dim ) GOTO 9000
	  answer = -answer
	ELSEIF ( e_given ) THEN

* longE - no change
	  IF ( idim .NE. x_dim ) GOTO 9000
	ELSEIF ( w_given ) THEN

* longW - convert	
	  IF ( idim .NE. x_dim ) GOTO 9000
	  answer = 360. - answer
	ENDIF

	status = merr_ok
	GOTO 9990

* errors
 9000	CALL TM_ERRMSG ( merr_syntax, status, 'TM_TRANSLATE_LAT_LONG',
     .	                 no_descfile, no_stepfile, 
     .			 string, no_errstring, *9990 )

 9990	RETURN
	END
