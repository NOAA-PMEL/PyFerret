	subroutine geochar (xlat, xlon, title)
C** 
C**    @(#)geochar.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Receives xlat, xlon as EPIC real numbers.  Returns formatted character
c  string title with the latitude and longitude (eg." 2N 110W", "10N  95W")
c  in 8 characters.
c
c  Programmed by N. Soreide, Apr 86.
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	character title*(*)
c
	if (xlat .gt. 0) then
	  write (title(1:4),100) ifix(xlat+.5001)
100	  format (i2,'N ')
	else
	  write (title(1:4),101) ifix(-xlat+.5001)
101	  format (i2,'S ')
	endif
	if (xlon .gt. 180.) then
	  write (title(5:8),102) ifix(360.-xlon+.499)
102	  format (i3,'E')
	else
	  write (title(5:8),103) ifix(xlon+.499)
103	  format (i3,'W')
	endif
	return
	end
