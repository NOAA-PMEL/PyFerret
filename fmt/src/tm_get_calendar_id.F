      INTEGER FUNCTION TM_GET_CALENDAR_ID (cal_name)

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
C     Return the calendar id associated with cal_name, a calendar name
C     Zero is returned if cal_name is not a valid calendar.
C     J Davison 10.99
C     V 1.0
C
C     0.00000000000000000000000000000000000000000000000000000000000000001

C 2/13/01  ACM  test for the calendar names: 
C 4/30/04       add 366_DAY, ALL_LEAP, and 365_DAY
C	"STANDARD" which is equivalent to 'GREGORIAN'
C	"COMMON_YEAR" and 365_DAY" for "NOLEAP"
C	"D360"  or "360" for "360DAY"
C	"366_DAY" for "ALL_LEAP"
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files, add disclaimer
* v744 11/2011 *acm* Issue 1906: If PROLEPTIC_GREGORIAN is given then 
*                  override mixed-gregorian calendar handling. 

      include 'calendar.decl'
      include 'calendar.cmn'

      integer         end,ndx
      character*255   test_string

      character*(*)   cal_name

*************************************************************************

C     MOVE INPUT STRING TO UPPER CASE
      call upnsquish (cal_name,test_string,end)

C  Allow STANDARD same calendar as GREGORIAN
      IF (test_string(1:3) .EQ. 'STA')  test_string = 'GREGORIAN'
      IF (test_string(1:3) .EQ. 'PRO')  test_string = 'PROLEPTIC_GREGORIAN'
      IF (test_string(1:3) .EQ. 'COM')  test_string = 'NOLEAP'
      IF (test_string(1:3) .EQ. '365')  test_string = 'NOLEAP'
      IF (test_string(1:3) .EQ. '366')  test_string = 'ALL_LEAP'
      IF (test_string(1:3) .EQ. 'D36')  test_string = '360DAY'
      IF (test_string(1:3) .EQ. '360')  test_string = '360DAY'

C     LOCATE NAME IN CALENDAR TABLE
      do 10 ndx = 1, max_calendars
         if (test_string(1:3) .eq. allowed_calendars(ndx)(1:3))
     .        goto 100
 10   end do

C     ERROR: CALENDAR NOT FOUND
      tm_get_calendar_id = 0
      goto 1000

C     CALENDAR FOUND, RETURN ID
 100  tm_get_calendar_id = ndx
 1000 return

      end
