	subroutine epictlim (xdate,xtime,xdat2,xtim2)
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
C**    @(#)epictlim.F	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Called by PLOT5's RDSEQ.  Input is starting xdate,xtime and
c  ending xdat2,xtim2 for EPIC time series.  Output is itmin,itmax
c  which is range of times for all time series currently in buffer.
c
c  Programmed by N. Soreide, May 86.
c
#ifdef unix
	include 'pplinc/AXISL.INC'
	include 'pplinc/TAXIS.INC'
	include 'pplinc/DATA.INC'
#else
	include 'pplinc:axisl.inc'
	include 'pplinc:taxis.inc'
	include 'pplinc:data.inc'
#endif
c
	if (linen .eq. 1) then
c
c  First series in buffer
c
	    call epictimc (xdate,xtime,itmin)
	    if (iautot .eq. 1) itstrt=itmin
	    call epictimc (xdat2,xtim2,itmax)
	else
c
c  Find time range minimum
c
	    read (itmin,100) xd,xt
100	    format (f6.0,f4.0)
	    if (xdate .lt. xd) then
	        call epictimc (xdate,xtime,itmin)
	    else if (xdate.eq.xd .and. xtime.lt.xt) then
		call epictimc (xdate,xtime,itmin)
	    endif
c
c  Find time range maximum
c
	    read (itmax,100) xd,xt
	    if (xdat2 .gt. xd) then
		call epictimc (xdat2,xtim2,itmax)
	    else if (xdat2.eq.xd .and. xtim2.gt.xt) then
		call epictimc(xdat2,xtim2,itmax)
	    endif
	endif
	return
	end
