	subroutine upcase (string,nc)
C** 
C**    @(#)upcase.F	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Converts the n characters in chrstr to upper case.
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
        character string(nc),temp
c
        do 200 i=1,nc
          istr = str$upcase (temp, string(i))
          string(i)=temp
200     continue
#endif
	RETURN
	END
