      FUNCTION LNBLK(STRING, NCHAR)
C** 
C**    @(#)lnonbl5.f	1.1    3/10/88
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
CC  THIS FUNCTION RETURNS THE CHARACTER POSITION
CC  OF THE LAST NON-BLANK CHARACTER IN THE STRING.
CC  CHARACTERS ARE NUMBERED STARTING WITH 1 AT THE
CC  LEFT OF THE FIRST WORD AND ARE PACKED TEN WORD
CC  WITH BLANK FILL. THE LAST WORD IS SCANNED FROM
CC  THE FAR RIGHT REGARDLESS OF WHETHER NCHAR INDICATES
CC  ONLY A PARTIALLY FULL WORD.
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER STRING*(*)
	DO 10 I=NCHAR,1,-1
	IF(STRING(I:I).NE.' ')GOTO 11
10	CONTINUE
	LNBLK=0
	RETURN
C
C	RSX-11M
C
C11	LNBLK=I-1
C
C	END RSX-11M
C
C	VAX/VMS
C
11	LNBLK=I
C
C	END VAX/VMS
C
      RETURN
      END
