	FUNCTION VARCK (HEADER, CHOICE, YCOLUMN)
C** 
C**    @(#)varck.f	1.1.1.1    7/30/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
C
C  READS EPIC HEADER TO FIND THE COLUMN NUMBER (YCOLUMN) OF THE
C  GENERIC VARIABLE CHOICE.  IF CHOICE IS NEGATIVE, NO MESSAGE
C  IS PRINTED ON OUTPUT (UNIT6) WHEN THE SELECTED VARIABLE IS NOT
C  FOUND ON THE DATA FILE.
C
C  EXTRACTED FROM M. VERSCHELL'S PLOTSETUP BY N SOREIDE, JUN 85.
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
	CHARACTER*(*) HEADER(8)
	LOGICAL VARCK
	INTEGER*4 CHOICE,YCOLUMN,ICNT,CODEVAL(0:8),VARCODE
c
C  CODEVAL setup
c    This identifies the variable code range a variable type can be
c    found in. The range is the number:number+9. If the number is 999
c    then the variable is a special case, and must be dealt with 
c    specially.
c
c  Generic variable codes are given here.  Routine will return column
c  number of the variable:
c
c                    Generic   
c                      Code    Description
c		     ------   -----------
c			0     Pressure
c			1     Temperature
c			2     Salinity
c			3     Sigma-T
c			4     Oxygen
c			5     Conductivity
c			6     U
c			7     V
c			8     Dynamic Height
c			9     Other
c
      DATA CODEVAL /0,20,40,999,60,50,999,999,10/
c
C
	ICABS=IABS(CHOICE)
	READ (HEADER(3)(79:80),101) NVARS
101	FORMAT (I2)
      DO 300 ICNT=1,NVARS
        READ(HEADER(5)(ICNT*4-3:ICNT*4),1030) VARCODE
 1030   FORMAT(I4)
        IF (ICABS .EQ. 3) THEN
          IF (VARCODE .EQ. 70) GOTO 400
        ELSE IF (ICABS .EQ. 6) THEN
          IF((VARCODE.EQ.320).OR.(VARCODE.EQ.322).OR.(VARCODE.EQ.324)
     >       .OR.(VARCODE.EQ.326).OR.(VARCODE.EQ.330)
     >       .OR.(VARCODE.EQ.332).OR.(VARCODE.EQ.334)
     >       .OR.(VARCODE.EQ.336).OR.(VARCODE.EQ.420)
     >       .OR.(VARCODE.EQ.340)
     >       .OR.(VARCODE.EQ.422)) GOTO 400
        ELSE IF (ICABS .EQ. 7) THEN
          IF((VARCODE.EQ.321).OR.(VARCODE.EQ.323).OR.(VARCODE.EQ.325)
     >       .OR.(VARCODE.EQ.327).OR.(VARCODE.EQ.331)
     >       .OR.(VARCODE.EQ.333).OR.(VARCODE.EQ.335)
     >       .OR.(VARCODE.EQ.337).OR.(VARCODE.EQ.421)
     >       .OR.(VARCODE.EQ.341)
     >       .OR.(VARCODE.EQ.423)) GOTO 400
        ELSE IF (ICABS .EQ. 9) THEN
          IF ((YCOLUMN .EQ. ICNT).AND.(VARCODE .NE. 0)) GOTO 400
        ELSE 
          IF((VARCODE.GE.CODEVAL(ICABS)).AND.
     >       (VARCODE.LE.(CODEVAL(ICABS)+9))) GOTO 400
        ENDIF
  300 CONTINUE
  310    IF (CHOICE .GT. 0) WRITE(6,1040) 
 1040 FORMAT(/' There is no match for this variable. Cast OMITTED.'/)
	VARCK = .FALSE.
	RETURN
c
  400 YCOLUMN = ICNT
	VARCK = .TRUE.
	RETURN
	END
