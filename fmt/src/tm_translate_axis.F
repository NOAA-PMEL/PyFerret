	SUBROUTINE TM_TRANSLATE_AXIS (string, idim, answer, status)
*
* convert an ascii representation of an axis limit into a floating point 
* representation using special format rules
*
* programmer - Mark Verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* possible formats for world coordinate values:
*   x coord:  #  or #E  or  #W  for East, East, and West respectively
*   y coord:  #  or #S  or  #N  for South, South, and North respectively
*   z coord:  #
*   t coord:  #  or dd-mmm-yyyy@hh:mm:ss  eg 14-JAN-1986@11:15
*
* revision 0.00 - 12/13/88 - from TRANSLATE_LIMITS by Steve Hankin
*
* Include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/gt_lib.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

* calling argument declarations:
	CHARACTER*(*)	string
	INTEGER		status, idim
	REAL*8		answer

* internal variable declarations:
	INTEGER*4	end_pos, dy, mnth, yr, hr, mnt, scnd
	INTEGER*4	TM_LENSTR1
	REAL*8		TM_SECS_FROM_BC

* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9) )

	end_pos = TM_LENSTR1( string )

	IF ( idim .NE. t_dim ) THEN
* /X /Y or /Z
	  CALL TM_TRANSLATE_LAT_LONG ( string(1:end_pos), idim, answer,
     .	                               status )
	  IF ( status .NE. merr_ok ) GOTO 9990		! already reported

	ELSE
* /T
* ... is it a timestep?
	  READ (string(1:end_pos),*,ERR=200) answer
	  GOTO 1000
* ... is it a date/time string? (store as seconds for further processing)
  200	  CALL TM_BREAK_DATE(string,dy,mnth,yr,hr,mnt,scnd,status)
	  IF ( status .NE. merr_ok ) GOTO 9990 		! Report in call routine
	  answer = -1.D0*TM_SECS_FROM_BC(dy,mnth,yr,hr,mnt,scnd,status)
	ENDIF

* successful completion
 1000	status = merr_ok

 9990	RETURN
	END
