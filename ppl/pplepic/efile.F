	subroutine efile (datafile,dat,leof)
C** 
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
