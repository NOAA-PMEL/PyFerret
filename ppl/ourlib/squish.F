      SUBROUTINE SQUISH(JSTRNG,JFIRST,NPOS)
C** 
C**    @(#)squish.f	1.1    3/10/88
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
C	Removes blank characters starting at position JFIRST
C	for the next NPOS characters and packs the characters
C	back into the original string.  Good for getting rid
C	of extra blanks.
C
C	JSTRNG -- character, input and output string
C	JFIRST -- INT, position to begin processing
C	NPOS -- INT, number of characters to process
C
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER*1 JSTRNG*(*),KLPAR,KRPAR,JBLNK,JCH
      LOGICAL REMOVE

      DATA KLPAR/'('/,KRPAR/')'/,JBLNK/' '/

      IF(NPOS.EQ.0)RETURN
      LAST=IABS(NPOS)+JFIRST-1
      REMOVE=.FALSE.
      IF(NPOS.LT.0)REMOVE=.TRUE.
      NEXT=JFIRST
      NPAREN=0

      DO 500 JPOS=JFIRST,LAST
	JCH=JSTRNG(JPOS:JPOS)
	JSTRNG(JPOS:JPOS)=JBLNK
      IF(JCH.NE.KLPAR)GO TO 100
      NPAREN=NPAREN+1
      GO TO 150

  100 IF(NPAREN.EQ.0)GO TO 200
      IF(JCH.NE.KRPAR)GO TO 300
      NPAREN=NPAREN-1
  150 IF(REMOVE)JCH=JBLNK
      GO TO 300

  200 IF(JCH.EQ.JBLNK)GO TO 500
  300 JSTRNG(NEXT:NEXT)=JCH
      NEXT=NEXT+1
  500 CONTINUE

      RETURN
      END
