	subroutine epictrd (ndata, di, ti, df, tf, dt, leof, prhead)
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
C**    @(#)epictrd.F	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Reads name of a time series data file from unit lun11 (an EPIC file or other
c  pointer file opened by subroutine epicopn).  If end-of-file encountered on
c  unit lun11,leof is set .true.. Opens data file on unit lun1 and reads header.
c  Prints header on output file if prhead is true, not if false.  Reads
c  number of variables and variable codes.  Returns nvars and varcd(i) in
c  common/comvar/.
c
c  Programmed by N Soreide, Jun 85.
c  Modified for other pointer file and variable info, NNS, Oct 85.
c  Modified from epicrd for time series data, NNS, Mar 86.
c
	logical leof,prhead
c
c Subroutine READHD
	character*80 theader(8)
	common /comthdr/ theader
c
c  Subroutine EPICRD 
	character*4 varcd(20)
	common/comvar/ nvars,varcd
	common/comefil/datafile
	common/comedat/dat
	character line*132, dat*3, datafile*132
	common /comerd/ line
c
c  Subroutine EPICOPN EPIC file indicator
	logical lepic
	common /comepl/ lepic
c
	common /epiclun/ lun1,lun11,lun21
c
	data nline /132/
c
c  Read data file name, open data file and read CTD header.
c
	if (lun1 .eq. 0) lun1=1
	if (lun11 .eq. 0) lun11=11
c
	leof=.false.
c
	if (lepic) then
		call efile (datafile, dat, leof)
		if (leof) return
	else
		read (lun11, 102, end=300) datafile
102		format (a)
	endif
c
#ifdef unix
	open (unit=lun1, form='unformatted', status='old', 
     *	file=datafile)
#else
	open (unit=lun1, form='unformatted', status='old', 
     *	readonly,file=datafile)
#endif
c
c  Check that this is time series data
c
	rewind lun1
	read (lun1) theader(1)
	read (lun1) theader(2)
	if (theader(2)(78:78) .ne. 'T') then
		write (6,104) theader(2)(78:78)
104		format (//' This is not time series data.',
     *		' Header type is ', a, ' - STOP')
		stop
	endif
	rewind lun1
c
c  Read header, get number of variables and variable codes.
c
	if (prhead) write (6,103) datafile(1:lenstr(datafile))
103	format (/ ' Data file name is ', a /)
c
	call readthd (lun1, 0, ndata, di, ti, df, tf, dt, prhead,
     *	.false., .false.)
	read (theader(3)(79:80), 602) nvars
602	format (i2)
	read (theader(5), 603) (varcd(i),i=1,nvars)
603	format (20a4)
	return
c
c  End-of-file encountered on list of data file names (EPIC file)
c
300	leof=.true.
	return
	end
