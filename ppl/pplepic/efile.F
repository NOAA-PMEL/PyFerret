	subroutine efile (datafile,dat,leof)
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
C**    @(#)efile.f	1.1.1.1    7/30/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Reads lines of a Time Series or CTD-type EPIC file and returns
c  the line itself (line), the data file specifications (datafile) and the
c  data type info (dat).  If eof is encountered, leof is set to true.
c
c  Programmed by N. Soreide, Apr 86.
c
	common /epiclun/ lun1,lun11,lun21
c
c  Internal COMMON block
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	common /cefile/ ndiski,ndiskf,ndiri,ndirf,nfili,nfilf,ifirst 
	logical leof,lrim
	character direct*20, filename*22, disk*4
	character line*132, dat*(*), datafile*(*)
	common /comerd/ line
	data nline /132/
#ifdef unix
	external init_cefile
#else
	data ifirst /1/
#endif
	data lrim/.true./
c
	if (lun11 .eq. 0) lun11=11
C
	if (.not. lrim ) then
	    read (lun11, 102, end=310) datafile
	    datafile=datafile(1:lenstr(datafile))
	    return
310	    leof = .true.
	return
	endif
c
210	read (lun11, 102, end=300)line
102	format (a)
c
	if (ifirst .eq. 1) then
		ndat = index (line(1:nline), 'DAT ')
		if (ndat .ne. 0) then
			read (lun11, 102) line
			read (lun11, 102) line
			dat(1:3)=line(ndat:ndat+2)
		endif
		ndiri=index(line(1:nline), 'DIRECT')
		if (ndiri .eq. 0) go to 210
		ndiski=index (line(1:nline), 'DISK')
		nfili = index (line(1:nline), 'FILE')
		read(lun11, 102) line(1:nline)
		read(lun11, 102) line(1:nline)
		nfilf=nline
		ifirst = 0
	endif
c
		if (index(line,'[') .eq. 0) goto 210
		filename(1:) = line(nfili:nfilf)
		nfirblk=index(filename(1:),' ')
		if (nfirblk .ne. 0) filename(1:)=filename(1:nfirblk)
		ndirf = index (line(ndiri:nline),']') + ndiri - 1
		direct(1:) = line(ndiri:ndirf)
		if (ndiski .eq. 0) then
			disk(1:) = 'DH:'
		else
			ndiskf = index (line(ndiski:nline),' ') + ndiski - 1
			disk(1:) = line (ndiski:ndiskf)
		endif
		nk=lenstr(disk)
		nd=lenstr(direct)
		nf=lenstr(filename)
c
c  Check for "[" at start of directory name
c
		if (direct(1:1) .ne. '[') go to 210
c
		datafile=disk(1:nk)//direct(1:nd)//filename(1:nf)
	return
300	if (ifirst .eq. 0) then
	    leof = .true.
	    ifirst = 1
	    return
	else
	    rewind lun11
	    lrim = .false.
	    read (lun11,102)datafile
	    datafile=datafile(1:lenstr(datafile))
	    return
	endif
	end

#ifdef unix
	block data init_cefile

	common /cefile/ ndiski,ndiskf,ndiri,ndirf,nfili,nfilf,ifirst 
	data ifirst /1/
	end
#endif
