	LOGICAL FUNCTION TRUE_OR_FALSE (string, status)

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
* check to see if the string represents a TRUE condition or a FALSE one
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420  4/25/96
*	7/96 - minor compiler bug fix -- eliminate unuser RETURN

* calling argument declarations:
	INTEGER		status
	CHARACTER*(*)	string

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* local parameter declarations
	INTEGER nyes, nno
	PARAMETER (nyes = 4,
     .		   nno  = 6)

* local variable declarations
	LOGICAL TM_DIGIT
	INTEGER	STR_MATCH, i
	REAL	value
	CHARACTER*8 yes_strings(nyes), no_strings(nno)

	DATA (yes_strings(i),i=1,nyes)/'TRUE','T','Y','YES'/
	DATA (no_strings(i),i=1,nno)/'FALSE','F','N','NO','BAD','MISSING'/
	
* initialize
	status = ferr_ok

* a blank string is a "false"
	IF ( string .EQ. ' ' ) GOTO 2000

* if the string is numerical then "0" is false, all other is true
	IF ( TM_DIGIT(string) ) THEN
	   READ ( string, *, ERR=5100) value
	   IF ( value .EQ. 0.0 ) THEN
	      GOTO 2000
	   ELSE
	      GOTO 1000
	   ENDIF

* if the string is non-numerical then try matching
	ELSE
	   i = STR_MATCH( string, yes_strings, nyes )
	   IF ( i .NE. atom_not_found ) GOTO 1000	! a valid TRUE
	   i = STR_MATCH( string, no_strings,  nno  )
	   IF ( i .NE. atom_not_found ) GOTO 2000	! a valid FALSE
	   GOTO 5100					! syntax err

	ENDIF

* return TRUE
 1000	TRUE_OR_FALSE = .TRUE.
	RETURN

* return FALSE
 2000	TRUE_OR_FALSE = .FALSE.
	RETURN


* error exit
 5100	TRUE_OR_FALSE = .FALSE.		! included for whiny OSF compiler
        CALL ERRMSG( ferr_syntax, status, string, *2000 )
	END
