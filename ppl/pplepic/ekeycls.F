	subroutine ekeycls
C** 
C**    @(#)ekeycls.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Closes EPICKEY file (Unit lun21).
c
c  Programmed by N. Soreide, Oct 85.
c
	common /epiclun/ lun1,lun11,lun21
	close (lun21)
	return
	end
