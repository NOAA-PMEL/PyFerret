	subroutine fixtim (it1,it2,itnew1,itnew2,itype)
C** 
C**    @(#)fixtim.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Input:
c	it1,it2 = input range of times in PLOT5's WHOI format (yyddmmhhmm)
c	itype   = PLOT5 axis type 0 even hours		ending     00
c                                 1 even days		ending   0000
c                                 3 even months		ending 010000
c  Output:
c	itnew1,itnew2 = range of times which suit the input axis type and
c                       include the input range of times (yymmddhhmm)
c
c  Uses subroutines TIME1 (sequence # <--> WHOI times and NUMDM1
c  (# days in month) from PLOT5's MILELIB.
c
c  Programmed by N. Soreide, May 86.

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
* Mod J Davison 11.21.95 For 4 digit years
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C 
C *acm* 4/22/02  replace ALL places lower than the desired round-off
C                with 00.  e.g. if 950115013303 is to be expressed
C                in even hours, make it 950115010000 not 950115010003

c
	character*(*) it1,it2,itnew1,itnew2
	character*14 cdum
c
	if (itype .eq. 3) then
cc acm	    itnew1=it1(1:4)//'010000'//it1(11:14)
	    itnew1=it1(1:4)//'01000000'//it1(13:14)

	    if (it2(5:10) .eq. '010000') then
		itnew2=it2
	    else
		dt=1440.
		nm=numdm1(it2)
		read (it2(5:6),*) id
		iseq=nm-id+2
cc		call time1 (1, it2(1:6)//'0000'//it2(11:14), iseq,
cc     1		                cdum, idum, itnew2, dt)
		call time1 (1, it2(1:6)//'000000'//it2(13:14), iseq,
     1		                cdum, idum, itnew2, dt)
	    endif

	else if (itype .eq. 1) then
cc	    itnew1=it1(1:6)//'0000'//it1(11:14)
	    itnew1=it1(1:6)//'000000'//it1(13:14)
	    if (it2(7:10) .eq. '0000') then
		itnew2=it2
	    else
		dt=1440.
		iseq=2
cc		call time1 (1, it2(1:6)//'0000'//it2(11:14), iseq,
cc     1		                cdum, idum, itnew2, dt)
		call time1 (1, it2(1:6)//'000000'//it2(13:14), iseq,
     1		                cdum, idum, itnew2, dt)
	    endif
	else if (itype .eq. 0) then
cc	    itnew1=it1(1:8)//'00'//it1(11:14)
	    itnew1=it1(1:8)//'0000'//it1(13:14)
	    if (it2(9:10) .eq. '00') then
		itnew2=it2
	    else
		dt=60.
		iseq=2
cc		call time1 (1, it2(1:8)//'00'//it2(11:14), iseq,
cc     1		                cdum, idum, itnew2, dt)
		call time1 (1, it2(1:8)//'0000'//it2(13:14), iseq,
     1		                cdum, idum, itnew2, dt)
	    endif
	endif
	return
	end

