	subroutine putch(c)
	character c*(*)
C** 
C**    @(#)putchsun.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
c	use sun fortran library putc call
c
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

#ifndef F90_NO_FPUTC
	ierr = fputc(0,c)
#else
	WRITE(0,999)c
999	format(a1)
#endif
	call flush(0)
	return
	end
