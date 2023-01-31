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
* Linux replacement for fdate/date routines.  
*
* kob 3.11.97

	SUBROUTINE FDATE(date)

        IMPLICIT NONE
	CHARACTER*(*) date

	CHARACTER*9 linux_date
	CHARACTER*2 month
	character*3 mmm
	CALL DATE_AND_TIME(linux_date)
	month = linux_date(5:6)
	if (month .eq. '01') then
		mmm = "Jan"
	else if (month .eq. '02') then
		mmm = "Feb"
	else if (month .eq. '03') then
		mmm = "Mar"
	else if (month .eq. '04') then
		mmm = "Apr"
	else if (month .eq. '05') then
		mmm = "May"
	else if (month .eq. '06') then
		mmm = "Jun"
	else if (month .eq. '07') then
		mmm = "Jul"
	else if (month .eq. '08') then
		mmm = "Aug"
	else if (month .eq. '09') then
		mmm = "Sep"
	else if (month .eq. '10') then
		mmm = "Oct"
	else if (month .eq. '11') then
		mmm = "Nov"
	else if (month .eq. '12') then
		mmm = "Dec"
	endif

	date = linux_date(7:8)//'-'//mmm//'-'//linux_date(3:4)

	return
	end

************************TODAY'S TIME******************************

	SUBROUTINE FTIME(time)
        IMPLICIT NONE

	CHARACTER*(*) time

	character*9 linux_date
	character*10 linux_time

	call DATE_AND_TIME(linux_date, linux_time)
	time = linux_time(1:2)//':'//linux_time(3:4)//':'//linux_time(5:6)

	return
	end

