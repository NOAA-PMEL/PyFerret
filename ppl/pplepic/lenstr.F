	function lenstr (string)
C** 
C**    @(#)lenstr.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Returns length of character string WITHOUT trailing blanks.
c  Returns length as zero if string is all blanks.
c
c  Programmed by N Soreide, Jun 85.
c
	character*(*) string
	l=len (string)
	do 200 i=l,1,-1
	  if (string(i:i).ne.' ') then
	    lenstr=i
	    return
	  endif
200	continue
	lenstr=0
	return
	end

