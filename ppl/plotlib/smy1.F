      SUBROUTINE SMY1(X,XFACT,YFACT,Y,YHI,YTIC,NSMTCY,
     .       xacross,hgrat,hcolor,hline) 
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
C**    @(#)smy1.f	1.1    3/10/88
* v554: *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks:
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
C     THIS SUBROUTINE PLOTS THE SMALL YTICS ON THE Y AXIS. 
C     THIS DIFERS FROM SMY IN THAT THE TICS ARE LARGER. 
C 

c arguments for making lines across the plot at tic marks.

        REAL xacross, dash1, dash2, dash3, dash4  ! for drawing the tic all the way across the plot
        REAL hgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        INTEGER hcolor(*)      ! pen number 
        LOGICAL hline          ! If true, do the lines across.
        LOGICAL solid          ! solid lines across at tics
        LOGICAL dashed         ! dashed lines across at tics

        solid = (hgrat(1,2).LT.0 .AND. hgrat(2,2).LT.0 .AND.  
     .            hgrat(3,2).LT.0 .AND.  hgrat(4,2).LT.0 )
        dashed = (hgrat(1,2).GT.0 .OR. hgrat(2,2).GT.0 .OR.  
     .            hgrat(3,2).GT.0 .OR.  hgrat(4,2).GT.0 )
        IF (dashed) THEN
           dash1 = hgrat(1,2)
           dash2 = hgrat(2,2)
           dash3 = hgrat(3,2)
           dash4 = hgrat(4,2)
        ENDIF

      IF(NSMTCY.LT.1) RETURN 
      SMTIC=YTIC/FLOAT(NSMTCY+1) 
      DO 10 I=1,NSMTCY 
          YTEMP=Y+FLOAT(I)*SMTIC 
          IF((YHI-YTEMP)/YTIC.GE.0) THEN
             CALL PLOTT7(X,YTEMP,XFACT, YFACT,1,-5) 
     
             IF ((solid.OR.dashed) .AND. hline) THEN ! make line across
                CALL COLOR(hcolor(2))
                IF (dashed) THEN
                   CALL DASHSZ(dash1,dash2,dash3,dash4)
                   CALL DASHES
                ENDIF
                CALL PLOT(X,ytemp,0,0) 
                CALL PLOT(xacross,ytemp,1,0)
                CALL PLOT(X,ytemp,0,0) 
                CALL COLOR(1)
                CALL VECTRS
             ENDIF
          ENDIF

   10     CONTINUE 
      RETURN 
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 ** 
      END 
