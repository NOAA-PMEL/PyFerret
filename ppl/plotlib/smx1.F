      SUBROUTINE SMX1(X,XHI,XTIC,NSMTCX,XFACT,YFACT,Y,
     .                yacross,vgrat,vcolor,vline) 
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
C**    @(#)smx1.f	1.1    3/10/88
C**
* v554: *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks:
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C 
C     THIS SUBROUTINE PLOTS THE SMALL XTICS ON THE X AXIS. 
C     IT DIFFERS FROM SMX IN THAT THE TICS ARE BIGGER. 
C 

        REAL yacross, dash1, dash2,dash3,dash4   ! for drawing the tic all the way across the plot
        REAL vgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        INTEGER vcolor(*)      ! pen number 
        LOGICAL vline          ! If true, do the lines across.
        LOGICAL solid          ! solid lines across at tics
        LOGICAL dashed         ! dashed lines across at tics

        solid = (vgrat(1,2).LT.0 .AND. vgrat(2,2).LT.0 .AND.  
     .            vgrat(3,2).LT.0 .AND.  vgrat(4,2).LT.0 )
        dashed = (vgrat(1,2).GT.0 .OR. vgrat(2,2).GT.0 .OR.  
     .            vgrat(3,2).GT.0 .OR.  vgrat(4,2).GT.0 )
        IF (dashed) THEN
           dash1 = vgrat(1,2)
           dash2 = vgrat(2,2)
           dash3 = vgrat(3,2)
           dash4 = vgrat(4,2)
        ENDIF

      IF(NSMTCX.LT.1) RETURN 
      SMTIC=XTIC/FLOAT(NSMTCX+1) 
      DO 10 I=1,NSMTCX 
          XTEMP=X+FLOAT(I)*SMTIC 
          IF((XHI-XTEMP)/XTIC.GE.0) THEN
             CALL PLOTT7(XTEMP,Y,XFACT,YFACT,1,-7) 

             IF ((solid.OR.dashed) .AND. vline) THEN ! make line across
                CALL COLOR(vcolor(2))
                IF (dashed) THEN
                   CALL DASHSZ(dash1,dash2,dash3,dash4)
                   CALL DASHES
                ENDIF
                CALL PLOT(xtemp,Y,0,0) 
                CALL PLOT(xtemp,yacross,1,0)
                CALL PLOT(xtemp,Y,0,0) 
                CALL COLOR(1)
                CALL VECTRS
             ENDIF
          ENDIF

   10 CONTINUE 
      RETURN 
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 ** 
      END 
