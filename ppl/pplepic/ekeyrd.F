	subroutine ekeyrd (varcd, labsml, label, comment)
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
C**    @(#)ekeyrd.F	1.1.1.1    7/30/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Reads EPICKEY file (Unit lun21) to get short graph label, full label and
c  comment for variable code varcd.
c
c  Programmed by N. Soreide, Oct 85.
c
	common /epiclun/ lun1,lun11,lun21
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C
C *kob* 1/03 - g77 port requires format of nX, rather than just X.  
C              Assuming 1x is proper
	character varcd*(*), labsml*(*), label*(*), comment*(*)
#ifdef unix
	character varrd*4
#endif
c
	if (lun21 .eq. 0) lun21=21
#ifdef unix
	rewind(lun21)
10	read (lun21, 100,err=200,end=200) varrd, labsml, label, comment
100	format (a4, 1x, a3, 1x, a25, 1x, a97)
	if(varrd.ne.varcd)goto 10
#else
        read (lun21, 100, key=varcd, err=200) labsml, label, comment
100     format (4x, x, a3, x, a25, x, a97)
#endif
	return
c
200	labsml(1:)=' '
	label(1:)=' '
	comment(1:)=' '
	return
	end
