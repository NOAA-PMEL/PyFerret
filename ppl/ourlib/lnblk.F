      FUNCTION LNBLK(STRING, NCHAR)
** 
C**    @(#)lnblk.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
CC  THIS FUNCTION RETURNS THE CHARACTER POSITION
CC  OF THE LAST NON-BLANK CHARACTER IN THE STRING.
CC  CHARACTERS ARE NUMBERED STARTING WITH 1 AT THE
CC  LEFT OF THE FIRST WORD AND ARE PACKED TEN WORD
CC  WITH BLANK FILL. NCHAR TELLS LNBLK WHERE TO START
CC  THE SEARCH FOR THE LAST NON BLANK CHARACTER
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER STRING*(*)
	DO 10 I=NCHAR,1,-1
	IF(STRING(I:I).NE.' ')GOTO 11
10	CONTINUE
	LNBLK=1
	RETURN
11	LNBLK=I
      RETURN
      END
