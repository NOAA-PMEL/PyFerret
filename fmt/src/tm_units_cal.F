	INTEGER FUNCTION TM_UNITS_CAL( inunit, cal_id, true_month )
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
* Reset the unit ID for a unit of month or year for non-gregorian calendars,
* according to calendar's year length.

*   v744 11/2011   *acm* If PROLEPTIC_GREGORIAN is given then override mixed-gregorian
*                  calendar handling. 
* 5/2020 Issue 1906: treat Proleptic-Gregorian identically to Gregorian

	include 'calendar.decl'
	include 'calendar.cmn'	
* Argument definitions
	LOGICAL true_month
	INTEGER inunit, cal_id

* Internal defitions
	INTEGER iun

	iun = inunit

C /calendar=/unit=year gets length of year in that calendar.
	IF (inunit .EQ. -6 .OR. inunit .EQ. -10) THEN  
	   IF (cal_id .EQ. d360) iun = -9
	   IF (cal_id .EQ. julian) iun = -12
	   IF (cal_id .EQ. noleap) iun = -11
	   IF (cal_id .EQ. all_leap) iun = -8
	ENDIF

C /units=months  is 1/12 of the length of the year

	IF (inunit .EQ. -5) THEN    ! 1/97: "-5" is "MONTHS"
	   IF (cal_id .eq. noleap) iun = -13
	   IF (cal_id .eq. julian) iun = -15
	   IF (cal_id .eq. d360  ) iun = -14
	   IF (cal_id .eq. all_leap) iun = -16
	   IF (true_month) THEN
	       IF (cal_id.eq.gregorian .or. cal_id.eq.proleptic) THEN
	          iun = -17
	       ELSE
	          iun = iun - 5
	      ENDIF
	   ENDIF
	ENDIF

	TM_UNITS_CAL = iun
	RETURN
	END
