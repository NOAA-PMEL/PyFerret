	subroutine init(nme,key,n)
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
C**    @(#)init.F	1.1    3/10/88
C** v552 *acm* 3/03 Longer strings: changes for multi-line labels.
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
c	vax/vms version of init
c
c	d. denbo   january 85
c
#ifdef unix
	character key(1)*5,nme(1)*81,line*2048,yn*3
	ic=iargc()
	if(ic.gt.0)then
	call getarg(1,line)
	else
	line=' '
	endif
	istp=2048
#else
	character key(1)*5,nme(1)*81,line*2048,cr*1,esc*1,yn*3
	byte buf(2048),crb,escb
	equivalence (line,buf(1)),(cr,crb),(esc,escb)
	integer*4 lib$get_foreign,istat
	data crb/13/,escb/27/
	istat=lib$get_foreign(line,,)
	istp=index(line,cr)
	if(istp.eq.0)istp=index(line,esc)
	if(istp.eq.0)istp=2048
#endif
	line(istp:istp)=' '
	is=1
100	k=index(line,'=')
	if(k.le.0.or.k.ge.istp)goto 200
	line(k:k)=' '
	do 10 i=1,n
	if(key(i).eq.line(is:k-1))goto 11
10	continue
	i=n+1
11	if=0
	do 20 j=k+1,istp
	if(line(j:j).eq.'[')then
	    if=1
	    ibrk=index(line(j:),']')+j-1
	    icma=index(line(j:),',')+j-1
	    if(icma.gt.ibrk)if=0
	endif
	if(line(j:j).eq.','.and.if.eq.1)then
	if=0
	goto 20
	endif

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
*       *jd* if stmt below placed entirely within ifdef for MAC 7.92 

#ifdef unix
	if(line(j:j).eq.','.or.line(j:j).eq.' '
     *	.or.line(j:j).eq.':')goto 21
#else
	if(line(j:j).eq.','.or.line(j:j).eq.' '
     *	.or.line(j:j).eq.'/')goto 21
#endif

20	continue
21	if(i.le.n)nme(i)=line(k+1:j-1)
	is=j+1
	goto 100
#ifdef unix
200	k=index(line,':')
#else
200	k=index(line,'/')
#endif
	if(k.le.0.or.k.ge.istp)return
	line(k:k)=' '
	if(line(k+1:k+1).eq.'-')then
	k=k+2
	yn='no'
	else
	k=k+1
	yn='yes'
	endif
	do 30 j=k,istp

*       *jd* as above

#ifdef unix
	if(line(j:j).eq.' '.or.line(j:j).eq.','
     *	.or.line(j:j).eq.':')goto 31
#else
	if(line(j:j).eq.' '.or.line(j:j).eq.','
     *	.or.line(j:j).eq.'/')goto 31
#endif
30	continue
31	do 40 i=1,n
	if(key(i).eq.line(k:j-1))goto 41
40	continue
	goto 200
41	nme(i)=yn
	goto 200
	end
