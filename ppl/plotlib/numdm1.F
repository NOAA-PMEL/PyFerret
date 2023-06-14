      FUNCTION NUMDM1(IT) 
C** 
C**    @(#)numdm1.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

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
* 	Mod J Davison 11.22.95 for 4 digit years
* 	Mod J Davison  2.6.96  fix bug re 400 year biz
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

*       J Davison 10.20.99   Modified to support multiple calendars.
*  *acm* 12/00 Change calendars to match NetCDF standard 
*  V5.70 *acm* 5/2004 Include the century in year iy, to get correct
*              evaluation of whether it is a leap year! (Bug 882)
* V68  *acm* 6/12  changes for double-precision ferret, single-precision pplus

      INCLUDE 'calendar.decl'
      INCLUDE 'calendar.cmn'

C     THIS FUNCTION RETURNS NO. OF DAYS IN CURRENT MONTH

      real*8 yeardays
      CHARACTER*(*) it

      call time3(it,iy,im,idd,ih,ix,isec,icen) 
      iy = icen* 100 + iy

      CALL tm_get_calendar_attributes (calendar_id, num_months, 
     .                 num_days, yeardays, month_names, days_in_month, 
     .                 days_before_month, month_by_day)

******************************************************************************
* HANDLE GREGORIAN CALENDAR FIRST
******************************************************************************
       if (calendar_id .eq. gregorian) then
         numdm1 = days_in_month(im) 

         if(im.eq.2.and.iy.eq.4*(iy/4)
     .        .and.mod(iy,100).ne.0)  numdm1=29	! centuries are not leap yrs

         if(im.eq.2.and.iy.eq.400*(iy/400)) 
     .        numdm1=29         ! every 400 years is tho...
      
******************************************************************************
* HANDLE JULIAN CALENDAR 
******************************************************************************
       else if (calendar_id .eq. julian) then
         numdm1 = days_in_month(im) 

         if(im.eq.2.and.iy.eq.4*(iy/4))  numdm1=29
      
******************************************************************************
* ALL OTHER SUPPORTED AND REGULAR CALENDARS
******************************************************************************
       else if (calendar_id .gt. gregorian .and. 
     .		calendar_id .le. max_calendars) then

         numdm1 = days_in_month(im) 
      else
         stop 'Illegal calendar in numdm1()'
      end if

      return 
      end 
