	subroutine epicvar
C** 
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
	include 'pplinc/CMRD.INC'
	include 'pplinc/CMRDL.INC'
	include 'pplinc/TAXIS.INC'
	include 'epicinc/COMEPV.INC'
	include 'epicinc/COMEPS.INC'
#else
	include 'pplinc:CMRD.INC'
	include 'pplinc:CMRDL.INC'
	include 'pplinc:TAXIS.INC'
	include 'epicinc:COMEPV.INC'
	include 'epicinc:COMEPS.INC'
#endif
	character ev1*5,ev2*5
c
	itflg=0
	ieflg=1
c
	if (index(label(1:n),'?') .ne. 0) then
c
c  Put out parameter list if parameter1 is ?
c
	    write (5,100)
100	    format (' Pressure        P'/
     1	                   ' Temperature     T'/
     2	                   ' Salinity        SAL'/
     3	                   ' Sigma-T         SIG'/
     4	                   ' Oxygen          OX'/
     5	                   ' Conductivity    CO'/
     6	                   ' U               U'/
     7	                   ' V               V'/
     8	                   ' Dynamic Ht      DYN'/
     9	                   ' Time            TIM'/
     1	                   ' Stick Plots     STK'/)
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
