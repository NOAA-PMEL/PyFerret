	SUBROUTINE CD_GET_CALENDAR_NAME (calname, ok_cal)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* Given a string with a calendar name, match the first 3 characters 
* with the list of allowed names or with the words STANDARD or 
* PROLEPTIC_GREGORIAN and return the standard name.

* Include files
        include 'calendar.decl'
        include 'calendar.cmn'
        include 'errmsg.parm'
        include 'xio.cmn_text'

* Argument declarations
	LOGICAL ok_cal
	CHARACTER*(*) calname

* Internal variables
	INTEGER STR_UPCASE, TM_LENSTR1, i, istat, ncal, slen
	CHARACTER*50 allcals

	INTEGER tt  ! nice short name
	EQUIVALENCE (tt,lunit_errors)

	istat = STR_UPCASE (calname, calname)

	IF (calname(1:3) .EQ. '360') calname = '360_DAY' 
	IF (calname .EQ. 'STANDARD') calname = 'GREGORIAN' 
	IF (calname .EQ. 'PROLEPTIC_GREGORIAN') calname = 'GREGORIAN' 
	IF (calname(1:6) .EQ. 'COMMON') calname = 'NOLEAP' 
	IF (calname(1:3) .EQ. '365') calname = 'NOLEAP' 
	IF (calname(1:3) .EQ. '366') calname = 'ALL_LEAP' 

	ok_cal = .FALSE.
	ncal = 1
	allcals = ' '

	DO 100 i = 1, mxcals
  111	   FORMAT (A9,' ')
	   slen = TM_LENSTR1( allowed_calendars(i)(1:9) )
	   write (allcals(ncal:ncal+9), 111) allowed_calendars(i)
	   ncal = ncal + 10

	   IF (calname(1:3) .EQ. allowed_calendars(i)(1:3) ) THEN
              ok_cal = .TRUE.
              calname = allowed_calendars(i)
	      GOTO 120
	   ENDIF
  100	CONTINUE
	CALL TM_NOTE('Valid calendars are '//allcals, tt)

  120	CONTINUE
	RETURN
	END
