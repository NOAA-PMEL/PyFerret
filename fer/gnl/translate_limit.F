	SUBROUTINE TRANSLATE_LIMIT
     .		(string, idim, subscript, answer, formatted, status)

* convert an ascii representation of a coordinate value into a floating point 
* representation using special format rules

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/11/86
* revision 0.1 - 3/27/87 - corrected bug that let ","," " or tab slip through
* revision 0.2 - 4/28/87 - allowed relative ("d") notation
* revision 0.3 - 3/25/88 - added date/time formatted input
* revision 0.4 -12/14/88 - corrected error detection
* revision 0.5 - 3/14/89 - return "formatted" flag
* V200:  6/15/89 - eliminated "d" notation and "relative" logic
*	11/30/89 - error check for comma in string
*	 2/16/90 - error check for slash in string
* Unix/RISC port - 3/12/91 - TM_DATE to replace READ(ERR=) testing date string
* V230:  6/19/92 - bug in error reporting using "risc_buff" fixed
* V300:  9/93 - allow "m" as in Z=15m to be acceptable
* V310: 4/94 - ignore blanks before or after the text
* V320: 1/95 - allow blanks in dates ("1-jan-1985 12:00")  TM_BREAK_DATE has
*		a thorough parser
* V420 (IBM/AIX port of V411): 10/5/95 - buff for "READ(var,*"

* possible formats:
* for subscript values ( subscript .EQ. .TRUE. ) :
*	#	- simple number
* for world coordinate values:
*   x coord:  #  or #E  or  #W     for East, West
*   y coord:  #  or #S  or  #N     for South, North
*   z coord:  #  
*   t coord:  #  or dd-mmm-$_@hh:mm:ss  eg 14-JAN-1986@11:15

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations:
	CHARACTER*(*)	string
	LOGICAL		subscript, formatted
	INTEGER		status, idim
	REAL*8		answer

* internal variable declarations:
	LOGICAL		TM_DIGIT, TM_DATE
	INTEGER		TM_LENSTR1, end_pos, pbad, i
	REAL*8		SECS_FROM_BC, given_answer
	CHARACTER	buff*32				! for AIX port 10/95

* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9) )

* initialize fixed length string for RISC
        risc_buff = string
	len_rbuff = MIN( size_rbuff, LEN(string) )

* imbedded comma, blank, slash or tab will go undetected as a READ syntax error
	pbad = INDEX( string, ',' )
	IF ( pbad .GT. 0 ) GOTO 5100
	pbad = INDEX( string, tab )
	IF ( pbad .GT. 0 ) GOTO 5100
	pbad = INDEX( string, '/' )
	IF ( pbad .GT. 0 ) GOTO 5100

* initialize
	end_pos = TM_LENSTR1( string )
	formatted = .FALSE.	! may be reset later

* special check for embedded blanks ignoring leading and following blanks 4/94
	DO 10 i = 1, end_pos
	   IF ( string(i:i) .NE. ' ' ) GOTO 11
 10	CONTINUE
	pbad = 1
	GOTO 5100    ! all blank string
 11	pbad = INDEX( string(i:end_pos), ' ' )
	IF ( pbad.GT.0 .AND. idim.NE.t_dim ) GOTO 5100   ! t_dim fallthru 1/95

* flag "D" ending as error as of version 2.00 of FERRET
	IF ( string(end_pos:end_pos) .EQ. 'd'
     .  .OR. string(end_pos:end_pos) .EQ. 'D' ) GOTO 5000

	buff = string(:end_pos)			! AIX port 10/95
	IF ( subscript ) THEN

* /I /J /K or /L
	   IF ( .NOT.TM_DIGIT(string(:end_pos)) ) GOTO 5000
	   READ ( buff,* , ERR=5000 ) given_answer
* make sure the value given is integer
	   answer = INT(given_answer)
	   IF ( ABS( answer-given_answer ) .GT. 0.0001 ) GOTO 5000

	ELSEIF ( idim .EQ. z_dim ) THEN    ! 9/93 addition to allow "m" at end
* /Z
	   formatted = string(end_pos:end_pos) .EQ.'m'
     .	          .OR. string(end_pos:end_pos) .EQ.'M'
	   IF ( formatted ) end_pos = end_pos - 1
	   READ ( buff,* , ERR=5000 ) answer

	ELSEIF ( idim .NE. t_dim ) THEN

* /X and /Y
	   CALL TRANSLATE_LAT_LONG ( string(1:end_pos), idim,
     .				     answer, formatted,status )
	   IF ( status .NE. ferr_ok ) GOTO 5900

	ELSE
* /T
* date/time string ?
           IF ( TM_DATE( string ) ) THEN
* ... time step cannot be evaluated until we know what axis this is on
* ... save number of seconds from B.C. as a negative to flag later translation
	      answer = -1.D0 * SECS_FROM_BC( string, status )
	      formatted = .TRUE.
	      IF ( status .NE. ferr_ok ) GOTO 5000
           ELSE
* ... read it as a time step value
	      READ ( buff,* , ERR=5000 ) answer
           ENDIF
        ENDIF

* successful completion
 1000	status = ferr_ok
	RETURN

* error exit
 5000	CALL ERRMSG( ferr_syntax, status, string, *5900 )
 5100	CALL ERRMSG( ferr_syntax, status,
     .		'"'//risc_buff(pbad:pbad)//'" is illegal in '//
     .		risc_buff(:len_rbuff), *5900 )
 5900	RETURN

	END
