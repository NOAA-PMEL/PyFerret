	subroutine epicrd (ncast, ndata, leof, prhead)
C** 
C**    @(#)epicrd.F	1.1.1.1    10/21/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Reads name of a data file from unit lun11 (an EPIC file or other pointer
C  file for CTD data opened by subroutine epicopn).  If end-of-file
c  encountered on unit lun11,  leof is set .true.. Opens data file on unit
c  lun1 and reads header.  Prints header on output file if prhead is true,
c  not if false.  Reads number of variables and variable codes.  Returns
c  nvars and varcd(i) in common/comvar/.
c
c  Programmed by N Soreide, Jun 85.
c  Modified for other pointer file and variable info, NNS, Oct 85.
c
	character direct*20, filename*14, datafile*132, disk*11
	logical leof,prhead
c
	common /epiclun/ lun1,lun11,lun21
c
c Subroutine READHD
	character*80 header(8),buffer(8)
	common /comhdr/ header
	common/dummy/buffer
c
c  Subroutine EPICRD 
	character*4 varcd(20)
	common/comvar/ nvars,varcd
	common/comefil/datafile
	character line*132, dat*3
	common /comerd/ line
c
c  Subroutine EPICOPN EPIC file indicator
	logical lepic
	common /comepl/ lepic
c
c  Read data file name, open data file and read CTD header.
c
	if (lun1 .eq. 0) lun1 = 1
	if (lun11 .eq. 0) lun11=11
c
	call efile(datafile,dat,leof)
	if (leof) goto 300
c
	call upcase(datafile,132)
#ifdef unix
	open (unit=lun1, form='unformatted', status='old',
     *	file=datafile)
#else
	call upcase(datafile,80)
	open (unit=lun1, form='unformatted', status='old',
     *	readonly,file=datafile)
#endif
c
   	ndfile=lenstr(datafile)
	if (prhead) write (6,103) datafile(1:ndfile)
103	format (/ ' Data file name is ', a /)
c
c  Read header, get number of variables and variable codes.
c
	call readhd (lun1, 0, ncast, ndata, 0, prhead, .false., .false.)
	read (header(3)(79:80), 602) nvars
602	format (i2)
	read (header(5), 603) (varcd(i),i=1,nvars)
603	format (20a4)
	return
c
c  End-of-file encountered on list of data file names (EPIC file)
c
300	continue
	return
	end
