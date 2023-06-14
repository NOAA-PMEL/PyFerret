	SUBROUTINE PARSEX(LINE,ILEN,X,INTIF,M,LABEL,N)
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
C**    @(#)parsex.F	1.2    5/26/88
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
C**		CALL PARSEX(LINE,ILEN,X,IF,M,LABEL,N)
C**
C**	Parameters:
C**		LINE -- Line to be parsed   (char*2048)
C**		ILEN -- Length of LINE
C**		X -- Numerical array
C**		INTIF -- Flag array
C**			=0 X() not changed
C**			=1 X() replaced
C**		M -- Length of numerical and flag array
C**		LABEL -- Character portion of LINE  (char*2048)
C**		N -- Length of LABEL
C**
C**	Description:
C**		Parses the numerical values in LINE placing the results
C**		in X and sets the flag array IF appropriatly.  The
C**		character information starting after the last valid 
C**		number in placed in LABEL.
C**
C**	History:
C**	    Written: 
C**		30-MAY-85	by Donald W. Denbo
C**	    Modified:
C**
C*
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C* v552 *acm* 3/03 Longer strings: changes for multi-line labels
* v580 *acm* 8/04 Change use of 256 to buflen+1, 
*                 where buflen is defined in cmndlen.inc
* V693+ *sh* 2/15 Allow _DQ_ to function like a double quote char

        INCLUDE 'cmndlen.inc'

	CHARACTER LABEL*(*),LINE*(*),FRMT*30
	DIMENSION X(*),INTIF(*)

*
*
*     *jd* variable IF renamed to INTIF for MAC 7.92
        INTEGER i                        ! ** TMAP mod 6/03 to init INTIF
C
	LOGICAL FIRST
C
C	Initialize output values
C
	LABEL = ' '
	N=0
        DO 90 I = 1, 20
	INTIF(I)=0
   90   CONTINUE
	M=1
	IF(ILEN.LE.0)THEN
	    M=0
	    RETURN
	ENDIF
	FIRST=.TRUE.
	I=1
C
C	Search for space or comma  (legal seperator)
C
100	IOLD=I
	ISPC=INDEX(LINE(I:),' ')+I-1
	IF(ISPC.EQ.I-1)ISPC=buflen+1
	ICMA=INDEX(LINE(I:),',')+I-1
	IF(ICMA.EQ.I-1) ICMA=buflen+1
	J=MIN0(ISPC,ICMA,ILEN+1)

* *acm 3/06 If i=j, see if what we have is a comma then space. 
*           Previously a space then comma was interpreted as
*           if it were two commas.  Check for just one comma
*           between elements, and can have any number of spaces.

        IF (i .EQ. j) THEN
           ICMA = INDEX(line(iold:i),',')
           IF (ICMA .NE. 0 .AND.
     .         INDEX(line(icma+1:i),',') .EQ. 0) THEN
              i = i + 1
              GOTO 100
           ENDIF
        ENDIF
C
C	Try to read a number,  if it fails put the rest into
C	LABEL if first is not true
C
	IF(I-J.EQ.0.OR.LINE(I:J-1).EQ.' ')THEN
	    INTIF(M)=0
	ELSE
C
C	Test for LINE(I:I) being +,-,., or 0-9.
C	That is a legal number
C
	    ITST=ICHAR(LINE(I:I))
	    IF(.NOT.(ITST.EQ.43.OR.ITST.EQ.45.OR.ITST.EQ.46
     *		.OR.(ITST.GE.48.AND.ITST.LE.57)))GOTO 1000
	    IWID=J-I
	    if(iwid.eq.1)then
		read(line(i:j-1),'(i1)',err=1000)jk
		x(m)=jk
	    else
	    WRITE(FRMT,999)IWID
999	    FORMAT('(F',I3.3,'.0)')
	    READ(LINE(I:J-1),FRMT,ERR=1000)X(M)
	    endif
	    INTIF(M)=1
	ENDIF
	FIRST=.FALSE.
	I=J
200	I=I+1
	IF(I.GT.ILEN)RETURN
	IF(LINE(I:I).EQ.' ')GOTO 200
	M=M+1
	GOTO 100
C
C	Error trying to read a number
C	clear the error and put the rest in LABEL
C
1000	continue
	IF(FIRST)THEN
	    FIRST=.FALSE.
	    I=J
	    M=M-1
	    GOTO 200
	ENDIF
	M=M-1
	I=IOLD

	IF(LINE(I:I).EQ.'"')THEN
	   I=I+1
	ELSEIF (LINE(I:I).EQ.'_'
     .   .AND. I+3 .LE. ILEN    ) THEN
	   IF (LINE(I:I+3).EQ.'_DQ_') I=I+4
	ENDIF
C
C get the label (argument)
C
	LABEL=LINE(I:)
	N=ILEN-I+1
	IF(LABEL(N:N).EQ.'"')THEN
	    LABEL(N:N)=' '
	    N=N-1
	ELSEIF (LABEL(N:N).EQ.'_'
     .	  .AND. N-3 .GE.1      ) THEN
	    IF (LABEL(N-3:N).EQ.'_DQ_') THEN
	      LABEL(N-3:N) = ' '
	      N=N-4
	    ENDIF
	ENDIF
	RETURN
	END
