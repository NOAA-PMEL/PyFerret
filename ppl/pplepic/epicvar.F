	subroutine epicvar
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
C**    @(#)epicvar.F	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Interprets PLOT5 EVAR command, which allows user to select EPIC variables
c  for plotting.
c
c  EVAR ?		Lists variable choices for user.
c  EVAR TIM,T		Selects x=time and y=temperature
c  EVAR			Selects x=time and y=first variable after time/date
c  EVAR STK     \_______
c  EVAR TIM STK /       Selects stick plots vs time
c  EVAR TIM,Cn          Selects x=time and y=variable in column n
c  EVAR TIM,En          Selects x=time and y=variable with EPICKEY
c                       variable code n.
c
c  Called in COMND.
c
c  Programmed by N. Soreide, May 86.
c
#ifdef unix
	include 'CMRD.INC'
	include 'CMRDL.INC'
	include 'TAXIS.INC'
	include 'COMEPV.INC'
	include 'COMEPS.INC'
#else
	include 'pplinc:CMRD.INC'
	include 'pplinc:CMRDL.INC'
	include 'pplinc:TAXIS.INC'
	include 'epicinc:COMEPV.INC'
	include 'epicinc:COMEPS.INC'
#endif
	character ev1*5,ev2*5
c
	print *, ' routine epicvar'
	itflg=0
	ieflg=1
c
	if (index(label(1:n),'?') .ne. 0) then
c
c  Put out parameter list if parameter1 is ?
c
	    write (5,100)
100	    format (' Pressure        P'/
     1              ' Temperature     T'/
     2              ' Salinity        SAL'/
     3              ' Sigma-T         SIG'/
     4              ' Oxygen          OX'/
     5              ' Conductivity    CO'/
     6              ' U               U'/
     7              ' V               V'/
     8              ' Dynamic Ht      DYN'/
     9              ' Time            TIM'/
     1              ' Stick Plots     STK'/)
c
c  If EVAR and no parameters, then use x=time, y=first variable after time
c
	else if (n .eq. 0) then
	    iepv1=9
	    iepv2=-1
	    if (lsctd) iepv1=-1
c
c  EVAR with parameters
c
	else
	    isp1=index(label(1:n),' ')
	    ico1=index(label(1:n),',')
	    if (isp1.eq.0 .and. ico1.eq.0) then
		write (5,101)
101		format (' EVAR command must have 2 parameters')
		return
	    else if (isp1 .eq. 0) then
		is=ico1
	    else if (ico1 .eq. 0) then
		is=isp1
	    else
		is=min0(isp1,ico1)
	    endif
	    ev1(1:)=label(1:is-1)
	    ev2(1:)=label(is+1:n)
	    call epicv (ev1,iepv1)
	    call epicv (ev2,iepv2)
	endif
c
c  If x-variable is STK, make x=TIM, and y=STK
c
	if (iepv1 .eq. 10) then
	    iepv1=9
	    iepv2=10
	endif
	return
	end
