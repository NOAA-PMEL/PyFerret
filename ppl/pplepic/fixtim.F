	subroutine fixtim (it1,it2,itnew1,itnew2,itype)
C** 
C**    @(#)fixtim.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Input:
c	it1,it2 = input range of times in PLOT5's WHOI format (yyddmmhhmm)
c	itype   = PLOT5 axis type 0 even hours		ending     00
c                                 1 even days		ending   0000
c                                 3 even months		ending 010000
c  Output:
c	itnew1,itnew2 = range of times which suit the input axis type and
c                       include the input range of times (yymmddhhmm)
c
c  Uses subroutines TIME1 (sequence # <--> WHOI times and NUMDM1
c  (# days in month) from PLOT5's MILELIB.
c
c  Programmed by N. Soreide, May 86.
c
	character*10 it1,it2,itnew1,itnew2,cdum
c
	if (itype .eq. 3) then
	    itnew1=it1(1:4)//'010000'
	    if (it2(5:10) .eq. '010000') then
		itnew2=it2
	    else
		dt=1440.
		nm=numdm1(it2)
		read (it2(5:6),*) id
		iseq=nm-id+2
		call time1 (1, it2(1:6)//'0000', iseq,
     1		                cdum, idum, itnew2, dt)
	    endif
	else if (itype .eq. 1) then
	    itnew1=it1(1:6)//'0000'
	    if (it2(7:10) .eq. '0000') then
		itnew2=it2
	    else
		dt=1440.
		iseq=2
		call time1 (1, it2(1:6)//'0000', iseq,
     1		                cdum, idum, itnew2, dt)
	    endif
	else if (itype .eq. 0) then
	    itnew1=it1(1:8)//'00'
	    if (it2(9:10) .eq. '00') then
		itnew2=it2
	    else
		dt=60.
		iseq=2
		call time1 (1, it2(1:8)//'00', iseq,
     1		                cdum, idum, itnew2, dt)
	    endif
	endif
	return
	end
