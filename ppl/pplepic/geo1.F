	function geo1 (lat)
C** 
C**    @(#)geo1.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Receives lat (character*8,9 with hemisphere designation).  Finds geo1
c  as a real number in EPIC lat/long notation.  Just like function GEO,
c  except GEO wants lat as character(8,9) not character*8,9.
c
c  Programmed by N. Soreide, Apr 86.
c
	character *(*) lat
	character clat(9)
	nc=lenstr(lat)
	read (lat,100) (clat(j),j=1,nc)
100	format (9a1)
	geo1 = geo (clat,nc)
	return
	end
