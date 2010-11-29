      subroutine gkinit
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
C**    @(#)gkinit.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
c
#ifdef unix
	include 'PLTCOM.DAT'
	include 'GKSCM1.INC'
#else
	include 'pplinc:PLTCOM.DAT'
	include 'pplinc:GKSCM1.INC'
#endif
	logical shrnkx,shrnky,first
c
	if(asize.gt.bsize)then
	    fcty=bsize/asize
	    fctx=1.0
	else
	    fctx=asize/bsize
	    fcty=1.0
	endif
	if(.not.gksopn)then
c
c	open and initialize gks
c
C Mod these three JD 3.7.91	    call gopks(errfil)
C                                   call gopwk(wsid,conid,wstype)
C	                            call gacwk(wsid)

	    call gopks(1)
	    call gopwk(wsid,0,0)
	    call gacwk(wsid)
	    gksopn=.true.
	    lpen=.not.pen
	endif
	call gswn(1,0.0,asize,0.0,bsize)
	call gselnt(1)
	call gsvp(1,0.0,fctx,0.0,fcty)
c
c	set up the size of the display window
c
	call gqwkc(wsid,ier,id,itype)
	call gqdsp(itype,ier,imeter,dmaxx,dmaxy,irx,iry)
c	call gqmds(itype,ier,imeter,dmaxx,dmaxy,irx,iry)

	if(wksaut)then
	    xln=asize*2.54e-5
	    yln=bsize*2.54e-5
	    first=.true.
200	    wkxmn=dmaxx-.0127-xln
	    wkymn=.00635
c
	    shrnkx=.false.
	    shrnky=.false.
	    if(wkxmn.lt.0)then
		wkxmn=0.0
		yln=yln*dmaxx/xln
		xln=dmaxx
		shrnkx=.true.
	    endif
	    if(wkymn+yln.gt.dmaxy)then
		wkymn=0.0
		xln=xln*dmaxy/yln
		yln=dmaxy
		shrnky=.true.
	    endif
	    if(first.and.shrnkx.and.shrnky)then
		first=.false.
		goto 200
	    endif
c
	    wkxmx=wkxmn+xln
	    wkymx=wkymn+yln
c
c	set the ndc window
c
	    call gswkwn(wsid,0.0,fctx,0.0,fcty)
c
c	set device (workstation) viewport
c
	    call gswkvp(wsid,wkxmn,wkxmx,wkymn,wkymx)
	else
c
c	if not auto wkxmn and wkxmx etc are fractions of dmaxx
c	and dmaxy
c
	    call gswkwn(wsid,0.0,fctx,0.0,fcty)
	    xmn=dmaxx*wkxmn
	    xmx=dmaxx*wkxmx
	    ymn=dmaxy*wkymn
	    ymx=dmaxy*wkymx
	    call gswkvp(wsid,xmn,xmx,ymn,ymx)
	endif
c
	return
	end
