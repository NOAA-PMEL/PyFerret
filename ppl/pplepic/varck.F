	FUNCTION VARCK (HEADER, CHOICE, YCOLUMN)
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
