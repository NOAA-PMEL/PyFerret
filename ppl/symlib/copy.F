	SUBROUTINE COPY(FROM,TO,IER)
C** 
C**    @(#)copy.f	1.2    5/26/88
C**
C**
C***********************************************************************
C**
C**		    PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**	Subprogram called:
C**		CALL COPY(FROM,TO,IER)
C**
C**	Parameters:
C**		FROM -- File to be copied (char*80)
C**		TO   -- File to be created (char*80)
C**		IER  -- Error code
C**			0 = success
C**			10 = spawn error
C**			20 = wait error
C**			2 = error - unlikly results to be expected
C**			4 = severe error - possible PIP abort
C**
C**	Description:
C**		COPY executes the MCR PIP command to copy a file.
C**
C**	History:
C**	    Written: 
C**		22-JAN-86	by Donald W. Denbo
C**	    Modified:
C**		30-MAY-86	by Donald W. Denbo
C**		  Changes made for VAX/VMS version
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER FROM*(*),TO*(*)
c
c	rsx-11m
c
c	INTEGER*2 WAIT,ERROR,FLAG,STATUS,IER
c	CHARACTER CMD*80
c	WRITE(CMD,999)TO,FROM
c999	FORMAT('PIP ',A30,'=',A30)
c	CALL SQUISH(CMD,5,61)
c	WAIT=0
c	FLAG=0
c	CALL TISPWN(CMD,WAIT,ERROR,FLAG,STATUS)
c	IF(ERROR.EQ.0.AND.STATUS.EQ.1)THEN
c	    IER=0
c	ELSE
c	    IER=10*ERROR+STATUS
c	ENDIF
c
c	end rsx-11m
c
c	vax/vms
c
	CHARACTER CMD*200	
	INTEGER IER
	WRITE(CMD,999)FROM,TO
999	FORMAT('$COPY ',A80,' ',A80)
#ifndef unix
	CALL LIB$SPAWN(CMD)
#endif
	IER=0
	RETURN
	END


