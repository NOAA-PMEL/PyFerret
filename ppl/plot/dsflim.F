	SUBROUTINE DSFLIM
C** 
C**    @(#)dsflim.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C	Subroutine called by REDSF to keep track of the maximum and
C	minimum times for the data
C
#ifdef unix
	INCLUDE 'pplinc/BIBO.COM'
	INCLUDE 'pplinc/AXISL.INC'
	INCLUDE 'pplinc/TAXIS.INC'
	INCLUDE 'pplinc/CMRD.INC'
	INCLUDE 'pplinc/DATA.INC'
#else
        INCLUDE 'pplinc:BIBO.COM'
        INCLUDE 'pplinc:AXISL.INC'
        INCLUDE 'pplinc:TAXIS.INC'
        INCLUDE 'pplinc:CMRD.INC'
        INCLUDE 'pplinc:DATA.INC'
#endif
C
	INTEGER*4 MTMIN,TSTART,TSTOP,TIME
	REAL STOP(3),TEMP(3)
C
	IF(LINEN.EQ.1)THEN
	    CALL MTMDY(P(STIME),IM,ID,IY)
	    WRITE(ITMIN,999)IY,IM,ID,IFIX(P(STIME+2))
999	    FORMAT(3I2.2,I4.4)
	    IF(IAUTOT.EQ.1)ITSTRT=ITMIN
	    TSTOP=MTMIN(P(STIME))+(P(NPTS)-1)*P(SRATE)
	    CALL MINMT(TSTOP,STOP)
	    CALL MTMDY(STOP,IM,ID,IY)
	    WRITE(ITMAX,999)IY,IM,ID,IFIX(STOP(3))
	ELSE
	    TSTART=MTMIN(P(STIME))
	    TSTOP=TSTART+(P(NPTS)-1)*P(SRATE)
	    CALL MINMT(TSTOP,STOP)
C
C	FIND TIME RANGE MINIMUM
C
	    READ(ITMIN,999)IY,IM,ID,IHR
	    CALL MDYMT(TEMP,IM,ID,IY)
	    TEMP(3)=IHR
	    TIME=MTMIN(TEMP)
	    IF(TSTART.LT.TIME)THEN
		CALL MTMDY(P(STIME),IM,ID,IY)
		WRITE(ITMIN,999)IY,IM,ID,IFIX(P(STIME+2))
	    ENDIF
	    READ(ITMAX,999)IY,IM,ID,IHR
	    CALL MDYMT(TEMP,IM,ID,IY)
	    TEMP(3)=IHR
	    TIME=MTMIN(TEMP)
	    IF(TSTOP.GT.TIME)THEN
		CALL MTMDY(STOP,IM,ID,IY)
		WRITE(ITMAX,999)IY,IM,ID,IFIX(STOP(3))
	    ENDIF
	ENDIF
	RETURN
	END
