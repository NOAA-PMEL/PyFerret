	subroutine epicsq (xd,xt,iseq)
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
C**    @(#)epicsq.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Input:
c  xd,xt = EPIC date, time words (type REAL) (xd=yymmdd and xt=hhmm).
c
c  Output:
c  iseq = Sequence number in minutes from a date which is approximately
c         the start of the century.  (type INTEGER*4)
c
c  Uses MILELIB function JDAY to perform calculations.  Used to get
c  EPIC time words into a sequence number for PPLUS.
c
c  Programmed by N. Soreide, Dec 86.
c
	integer*4 jday,iseq
c
	idate=xd+.001
	itime=xt+.001
c
	iy=idate/10000
	im1=idate-iy*10000
	im=im1/100
	id=im1-im*100
	ihr=itime/100
	imin=itime-ihr*100
c
	iseq = jday (iy, im, id, ihr, imin)
c
	return
	end

