	subroutine varkey (theader, ikey, jy)
C** 
C**    @(#)varkey.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Locates EPICKEY variable code ikey in EPIC header theader and returns
c  the column number in variable jy.  If the variable code is not found,
c  jy=0.
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	character*(*) theader(8)
c
	read (theader(3)(79:80),*) nvars
c
	do 200 i=1,nvars
	    iend=i*4
	    read (theader(5)(iend-3:iend),*) ivar
	    if (ivar .eq. ikey) goto 210
200	continue
	jy=0
	return
c
210	jy=i
	return
	end
