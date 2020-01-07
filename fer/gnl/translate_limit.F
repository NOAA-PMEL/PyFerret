	SUBROUTINE TRANSLATE_LIMIT
     .		(string, idim, subscript, answer, formatted, cal_id, 
     .		status)

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
* V500 *sh* - 6/99 check for quotations around text (e.g. date strings)
* V53 *acm* - 3/01 calendar axes:  pass in calendar ID
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead 
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction.
* V693+ *sh* 2/15 - added support for _DQ_ quotations
* V72   *acm* 11/16 - improve error message for invalid date strings
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* v74  *acm* 3/18  Issue 1856, changes to allow for timesteps in fractional seconds
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd
* V751 *acm*  5/19 Minor cleanup.  No longer need TM_DATE.  Report time-string translation
*                  errors for F or T axis
* V76  *acm*  1/20 Ticket 1963: part of issue was non-reported parsing error evaluating 
*                  the argument to /T= in LET/T="title"
* v76  1/2020 *acm* Getting error-reporting correct, logical time_err is set when
*                   an error was reported when translating iso-times and time doesn't 
*                   exist on the given calendar: no need for more messages

* possible formats:
* for subscript values ( subscript .EQ. .TRUE. ) :
*	#	- simple number
* for world coordinate values:
*   x coord:  #  or #E  or  #W     for East, West
*   y coord:  #  or #S  or  #N     for South, North
*   z coord:  #  
*   t coord:  #  or dd-mmm-$_@hh:mm:ss  eg 14-JAN-1986@11:15

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'tmap_errors.parm'
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'xrisc.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'

* calling argument declarations:
	CHARACTER*(*)	string
	LOGICAL		subscript, formatted
	INTEGER		status, idim
	REAL*8		answer

* internal variable declarations:
	LOGICAL		TM_DIGIT, TM_DATE_OK, TM_DATE, is_t, date_str
	INTEGER		TM_LENSTR1, end_pos, start_pos, pbad, i, 
     .                  slen, j, cal_id, dot
	REAL*8		SECS_FROM_BC, given_answer, frac
	CHARACTER	buff*60				! for AIX port 10/95

* internal parameter declarations:
	CHARACTER*1	tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif

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

* (6/99) check for enclosing quotes
	DO 4 i = 1, end_pos
	   IF (string(i:i) .EQ. '"') THEN
	      start_pos = i + 1
	      GOTO 6
	   ELSEIF ( string(i:i) .EQ. '_'
     .	      .AND. i+3 .LT. len_rbuff ) THEN
	      IF (string(i:i+3) .EQ. "_DQ_") THEN
	         start_pos = i + 4
	         GOTO 6
	      ENDIF
	   ENDIF
 4	CONTINUE
	start_pos = 1
 6	IF (start_pos .GT. 1) THEN
	   DO 8 i = end_pos, start_pos, -1
	      IF (string(i:i) .EQ. '"') THEN
	         end_pos = i - 1
	         GOTO 9
	      ELSEIF ( string(i:i) .EQ. '_'
     .	      .AND. i+3 .LE. len_rbuff ) THEN
	         IF (string(i:i+3) .EQ. "_DQ_") THEN
	            end_pos = i - 1
	            GOTO 9
	         ENDIF
	      ENDIF
 8	   CONTINUE
	ENDIF
 9	IF (start_pos .GT. end_pos) GOTO 5000

* special check for embedded blanks ignoring leading and following blanks 4/94
	DO 10 i = start_pos, end_pos
	   IF ( string(i:i) .NE. ' ' ) GOTO 11
 10	CONTINUE
	pbad = 1
	GOTO 5100    ! all blank string
 11	pbad = INDEX( string(i:end_pos), ' ' )
	IF ( pbad.GT.0 .AND. idim.NE.t_dim ) GOTO 5100   ! t_dim fallthru 1/95

* flag "D" ending as error as of version 2.00 of FERRET
	IF ( string(end_pos:end_pos) .EQ. 'd'
     .  .OR. string(end_pos:end_pos) .EQ. 'D' ) GOTO 5000

	buff = string(i:end_pos)		! AIX port 10/95
	slen = end_pos - i + 1
	IF ( subscript ) THEN

* /I /J /K or /L
	   IF ( .NOT.TM_DIGIT(string(i:end_pos)) ) GOTO 5000
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

	ELSEIF ( idim .EQ. x_dim .OR. idim .EQ. y_dim .OR. idim .EQ. e_dim) THEN

* /X and /Y and /E
	   CALL TRANSLATE_LAT_LONG ( buff(:slen), idim,
     .				     answer, formatted,status )
	   IF ( status .NE. ferr_ok ) GOTO 5900

	ELSE
* /T or /F
* date/time string ?
	   time_err = .FALSE.
	   IF ( TM_DATE_OK ( buff , cal_id)  ) THEN
* ... time step cannot be evaluated until we know what axis this is on
* ... save number of seconds from B.C. as a negative to flag later translation 

	      slen = TM_LENSTR1( buff )
	      dot = INDEX( buff(:slen), '.' )
	      frac = 0.
	      IF (dot .GT. 0) THEN 
		 READ (buff(dot:slen), *) frac
	         slen = dot-1
	      ENDIF
	      answer = -1.D0 * (SECS_FROM_BC( buff(:slen), cal_id, 
     .                                       status ) + frac)
	      formatted = .TRUE.
	      IF ( status .NE. ferr_ok ) THEN
	         slen = TM_LENSTR1(string)
		 buff = string(:slen)//' invalid date or not valid in this calendar'
		 GOTO 5200
	      ENDIF
           ELSE
* ... read it as a time step value
	      READ ( buff,* , ERR=5000 ) answer
           ENDIF
        ENDIF

* successful completion
 1000	status = ferr_ok
	RETURN

* error exit
 5000	IF ( (idim.EQ.t_dim .OR. idim.EQ.f_dim) .AND. time_err) THEN
	   status = ferr_syntax
	   time_err = .FALSE.
	   RETURN  ! already reported any error
	ELSE
	   CALL ERRMSG( ferr_syntax, status, string, *5900 )
	ENDIF
 5100	CALL ERRMSG( ferr_syntax, status,
     .		'"'//risc_buff(pbad:pbad)//'" is illegal in '//
     .		risc_buff(:len_rbuff), *5900 )
 5200	CALL ERRMSG( ferr_syntax, status, buff, *5900)
 5900	RETURN

	END
