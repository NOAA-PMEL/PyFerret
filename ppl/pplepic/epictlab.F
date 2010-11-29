	subroutine epictlab
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
C**    @(#)epictlab.F	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Reads PLOT5 x- and y-axis labels from EPIC key file for EPIC time series
c  data.  Also extracts plot title from time series header:  "description"
c  at "locaion (depth zz m)", where description is descript field if it's
c  non-blank, project/experiment/mooring otherwise.
c
c  Sets PPLUS symbols for depth, lat, long, comment lines from header,
c  xlab, ylab, project, mooring, descript from header.
c
c  Sets PPLUS symbols for CTD report-generator graph title with
c  info from first 4 lines of CTD header.
c
c  Programmed by N. Soreide, May 86.
c  Modified to set PPLUS symbol PPL$EPIC_DATAFILE(n), NNS, Aug 87.
c  Modified for CTD report-generator, NNS, Dec 87.
c
#ifdef unix
	include 'PARAMPL5.DAT'
	include 'COMDATF.INC'
	include 'COMEPV.INC'
	include 'COMEPS.INC'
	include 'COMVAR.INC'
	include 'COMTHDR.INC'
	include 'LABELS.INC'
	include 'DATA.INC'
#else
	include 'pplinc:parampl5.dat'
	include 'epicinc:comdatf.inc'
	include 'epicinc:comepv.inc'
	include 'epicinc:comeps.inc'
	include 'epicinc:comvar.inc'
	include 'epicinc:comthdr.inc'
	include 'pplinc:labels.inc'
	include 'pplinc:data.inc'
#endif
	character labsml*3,comment*97,cloc*8,sym*30
	character chalat*8,chalon*9,chapro*10,chaexp*5,chamoo*5
	character chadep*7
	character ctdcast*19,ctddate*14,ctdtime*13,ctdlat*12,ctdlon*14
	character cnumber*3,dnumber*3
	equivalence (chalat,theader(1)(6:13)),
     1	(chalon,theader(1)(22:30)),
     1	(chapro,theader(2)(36:45)),
     1	(chaexp,theader(2)(48:52)),
     1	(chamoo,theader(1)(48:52)),
     1	(chadep,theader(1)(39:45))
	equivalence (ctdcast,theader(1)(1:19)),
     1	(ctddate,theader(1)(23:36)),
     1	(ctdlat,theader(2)(1:12)),
     1	(ctdlon,theader(2)(15:28)),
     1	(ctdtime,theader(1)(40:52))
c
c  Read x- and y-axis labels from EPIC key file.
c
	call ekeyopn
	if (jx .gt. 0) then
	    call ekeyrd (varcd(jx),labsml,xlab,comment)
	    write (sym,100) 'XLAB',linen
100	    format ('*PPL$EPIC_', A, '(', I2.2, ')')
	    nchars=lenstr(xlab)
	    call putsym(sym,xlab(1:nchars),nchars,ier)
	endif
	if (jy .gt. 0) then
	    call ekeyrd (varcd(jy),labsml,ylab,comment)
	    write (sym,100) 'YLAB',linen
	    nchars=lenstr(ylab)
	    call putsym(sym,ylab(1:nchars),nchars,ier)
	endif
	call ekeycls
c
c  Read Graph Title and set PPLUS symbols
c
	if (lsctd) then
	    write (sym,100) 'LATITUDE',linen
	    call putsym(sym,ctdlat,12,ier)
	    write (sym,100) 'LONGITUDE',linen
	    call putsym(sym,ctdlon,14,ier)
	    write (sym,100) 'CAST',linen
	    call putsym(sym,ctdcast,19,ier)
	    write (sym,100) 'DATE',linen
	    call putsym(sym,ctddate,14,ier)
	    write (sym,100) 'TIME',linen
	    call putsym(sym,ctdtime,13,ier)
c
	    write (sym,100) 'CTD1HD',linen
	    call putsym(sym,theader(1)(1:52),52,ier)
	    write (sym,100) 'CTD2HD',linen
	    call putsym(sym,theader(2)(1:52),52,ier)
	    write (sym,100) 'CTD3HD',linen
	    call putsym(sym,theader(3)(1:52),52,ier)
	    write (sym,100) 'CTD4HD',linen
	    call putsym(sym,theader(4)(1:52),52,ier)
c
c  Get cast number into a PPLUS global symbol.  Strip blanks and
c  leading zeros from the cast number.
c
	    cnumber(1:3)=theader(1)(17:29)
	    if (cnumber(1:1) .eq. '0') then
		if (cnumber(2:2) .eq. '0') then
		    dnumber(1:)=cnumber(3:3)
		    idnum=1
		else
		    dnumber(1:)=cnumber(2:3)
		    idnum=2
		endif
	    else
		dnumber=cnumber
		idnum=3
	    endif
	    call stripit(dnumber(1:idnum),idnum,' ',cnumber,icnum)
	    if (icnum .eq. 0) then
		cnumber(1:1)=' '
		icnum=1
	    endif
	    write (sym,100) 'CASTNUM',linen
	    call putsym(sym,cnumber(1:icnum),icnum,ier)
	else
	    ipro=lenstr(chapro)
	    iexp=lenstr(chaexp)
	    imoo=lenstr(chamoo)
	    idep=lenstr(chadep)
	    write (sym,100) 'LATITUDE',linen
	    call putsym(sym,chalat,8,ier)
	    write (sym,100) 'LONGITUDE',linen
	    call putsym(sym,chalon,9,ier)
	    write (sym,100) 'PROJECT',linen
	    call putsym(sym,chapro,10,ier)
	    write (sym,100) 'EXPERIMENT',linen
	    call putsym(sym,chaexp,5,ier)
	    write (sym,100) 'MOORING',linen
	    call putsym(sym,chamoo,5,ier)
	    write (sym,100) 'DEPTH',linen
	    call putsym(sym,chadep,7,ier)
	    write (sym,100) 'COMMENT_DATA',linen
	    call putsym(sym,theader(6),lenstr(theader(6)),ier)
	endif
	write (sym,100) 'COMMENT_FIRST',linen
	call putsym(sym,theader(7),lenstr(theader(7)),ier)
	write (sym,100) 'COMMENT_SECOND',linen
	call putsym(sym,theader(8),lenstr(theader(8)),ier)
	write (sym,100) 'DATAFILE',linen
	call putsym(sym,datafile,lenstr(datafile),ier)
C
	if (lsctd) then
	    lab1(1:)=ctdcast//' '// ctddate//' '//
     1	    ctdlat//' '//ctdlon
	else
	    xlat=geo1(theader(1)(5:13))
	    xlon=geo1(theader(1)(22:30))
	    call geochar(xlat,xlon,cloc)
	    nd=lenstr(theader(4)(1:26))
	    write (sym,100) 'DESCRIPT',linen
	    call putsym(sym,theader(4)(1:26),26,ier)
	    if (nd.eq.0) then
c
c  descript is blank, so use proj/exp/moorname where these aren't blank.
c
		if (ipro .eq. 0) then
		    nlab=0
		else
		    lab1(1:22)=chapro
		    nlab=lenstr(lab1(1:22))
		endif
		if (iexp .ne. 0) then
		    lab1(nlab+1:22)=' / '//chaexp(1:iexp)
		    nlab=lenstr(lab1(1:22))
		endif
		if (imoo .ne. 0) then
		    lab1(nlab+1:22)=' / '//chamoo(1:imoo)
		endif
		nlab=lenstr(lab1(1:22))
	    else
c
c  use descript for title
c
		lab1 (1:26)=theader(4)(1:26)
		nlab=nd
	    endif
	    lab1(nlab+1:)=' at ' //
     1	    cloc // ' (' // theader(1)(33:45)
     1	    // ')'
	    lab1(1:)='@TR'//lab1
	endif
	return
	end
