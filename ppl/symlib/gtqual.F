	SUBROUTINE GTQUAL(LINE,ILEN,IER,IPOS)
C** 
C**    @(#)gtqual.F	1.2    5/26/88
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
C**		CALL GTQUAL(LINE,ILEN,IER,IPOS)
C**
C**	Parameters:
C**		LINE -- Input line containing qualifiers (char*255)
C**		ILEN -- Length of command portion
C**		IER  -- Error code =0 no error
C**       =3 qualifier not found
C**		IPOS -- Position of bad qualifier
C**
C**	Description:
C**		GTQUAL parses the command and qualifiers and sets the
C**		appropriate flag in the switch common blocks
C**
C**	History:
C**	    Written: 
C**		2-AUG-86	by Donald W. Denbo
C**	    Modified:
C**
C**
#ifdef unix
	INCLUDE 'cmdinc/SWITCH.INC'
#else
	INCLUDE 'cmdinc:SWITCH.INC'
#endif
	CHARACTER LINE*255
	LOGICAL TEST
C
C	SET DEFULTS
C
	IER=0
	IDONE=LNBLK(LINE,255)
C       Mod to allow Ferret to work in Unix as is -- using "/" for
C       qualifiers -- test mode only J Davison 3.22.91  3 ifdefs commented out
C*JD*#ifdef unix
C	ILEN = INDEX(LINE,':')-1
C#else
	ilen = index(line,'/')-1
C#endif
	IF(ILEN.LE.0)ILEN=IDONE
	DO 10 I=1,NFLAGS
10	FLAG(I)=DEFLTS(I)
C
C	PARSE LINE
C
	N=0
	IS=1
C*JD*#ifdef  unix
C20	ISLSH = INDEX(LINE(IS:),':')+IS-1
C#else
20	islsh = index(line(is:),'/')+is-1
C#endif
	IF(ISLSH.GE.IS)THEN
C
C	QUALIFIER FOUND
C
C*JD*#ifdef unix
C	    IEND = INDEX(LINE(ISLSH+1:),':')+ISLSH
C#else
	    iend = index(line(islsh+1:),'/')+islsh
C#endif
	    IF(IEND.LE.ISLSH)IEND=IDONE+1
	    TEST=LINE(ISLSH+1:ISLSH+2).NE.'NO'
	    IF(TEST)THEN
		ISLSH=ISLSH+1
	    ELSE
		ISLSH=ISLSH+3
	    ENDIF
	    LLEN=IEND-ISLSH
C
C	CHECK FOR LEGAL QUALIFIER
C
	    DO 30 J=1,NFLAGS
	    IF(LLEN.LT.SMIN(J))GOTO 30
	    IF(LINE(ISLSH:IEND-1).EQ.SWITCH(J)(1:LLEN))GOTO 31
30	    CONTINUE
	    IER=3
	    IPOS=ISLSH
	    RETURN
31	    FLAG(J)=TEST
	    IS=IEND
	    GOTO 20
	ENDIF
	RETURN
	END
