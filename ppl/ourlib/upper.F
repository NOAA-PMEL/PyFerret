	SUBROUTINE UPPER(STRING,NC)
C** 
C**    @(#)upper.F	1.1    3/10/88
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
c	converts the string to upper case
C
c	STRING -- character*80, input string
C	nc -- length of string
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

#ifdef unix
	CHARACTER STRING*(*),CHR(26)
	DATA CHR/'A','B','C','D','E','F','G','H','I',
     1	'J','K','L','M','N','O','P','Q','R','S','T',
     2	'U','V','W','X','Y','Z'/
	AA=ICHAR('a')
	DO 100 I=1,NC
	IF(STRING(I:I).GE.'a'.AND.STRING(I:I).LE.'z')THEN
	K=ICHAR(STRING(I:I))-AA+1
	STRING(I:I)=CHR(K)
	ENDIF
100	CONTINUE
#else
	CHARACTER STRING*80
	INTEGER*4 STR$UPCASE,ISTAT
	ISTAT = STR$UPCASE(STRING(1:NC),STRING(1:NC))
	IF(.NOT.ISTAT)CALL SYS$EXIT(%VAL(ISTAT))
#endif
	RETURN
	END
