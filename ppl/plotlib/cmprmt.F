	SUBROUTINE CMPRMT(PMT,N)
C** 
C**    @(#)cmprmt.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	COMMON/CPFTN5/PRMPT
	CHARACTER PRMPT*40,PMT*(*),FRMT*12
	IF(N.NE.0)THEN
		WRITE(FRMT,999)N
999		FORMAT('(A3,A',I2.2,',A4)')
		WRITE(PRMPT,FRMT)'('' ',PMT,''',$)'
	ELSE
		PRMPT='('' *'',$)'
	ENDIF
	RETURN
	END
