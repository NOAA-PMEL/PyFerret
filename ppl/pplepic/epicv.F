	subroutine epicv (ev,iev)
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
C**    @(#)epicv.F	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Translates character variable memonics in the PLOT5 EVAR command into
c  generic variable codes (integer) used in EPIC subroutine VARCK.
c
c  If the input variable memonic is unrecognized, then the character 
c  EVAR argument could contain a numeric column number.  If this is so,
c  then iev is set equal to this column number + 10000, to flag it as
c  a colunn number and not a generic integer variable code.
c
c  If numeric data can not be read from the character EVAR argument,
c  then the output generic code will be -1.
c
c  Programmed by N. Soreide, May 86.
c
#ifdef unix
	include 'TAXIS.INC'
#else
	include 'pplinc:taxis.inc'
#endif
	character *(*) ev
c
	iev=-1
	if (index (ev,'STK') .ne. 0) then
		iev=10
	else if (index (ev,'TIM') .ne. 0) then
		iev=9
		itflg=1
	else if (index (ev,'DYN') .ne. 0) then
		iev=8
	else if (index (ev,'SIG') .ne. 0) then
		iev=3
	else if (index (ev,'SAL') .ne. 0) then
		iev=2
	else if (index (ev,'OX') .ne. 0) then
		iev=4
	else if (index (ev,'CO') .ne. 0) then
		iev=5
	else if (index (ev,'P') .ne. 0) then
		iev=0
	else if (index (ev,'T') .ne. 0) then
		iev=1
	else if (index (ev,'U') .ne. 0) then
		iev=6
	else if (index (ev,'V') .ne. 0) then
		iev=7
c
c  See if character input is of form "Cn" where n is the column number 
c  (add 10000 to flag it as a column number) or of the form "En" where
c  n is the variable code from the EPICKEY file (add 20000 to flag it
c  as an EPICKEY variable code).
c
c       If the read is unsuccessful, then set the code to unknown (-1).
c
        else if (index(ev,'C') .ne. 0) then
            ichev=lenstr(ev)
            read (ev(2:ichev),*,err=100) iev
            iev=iev+10000
        else if (index(ev,'E') .ne. 0) then
            ichev=lenstr(ev)
            read (ev(2:ichev),*,err=100) iev
            iev=iev+20000
	else
c
c       If the character input to EVAR is unrecognized at this point, try
c       to read the column number from the character input.  If this is 
c       is successful, add 10000 to the column number to flag it as
c       a column number and not a generic integer variable code.
c
c       If the read is unsuccessful, then set the code to unknown (-1).
c
	    ichev=lenstr(ev)
	    read (ev(1:ichev),*,err=100) iev
	    iev=iev+10000
	endif
	return
c
100	iev=-1
	return
	end
