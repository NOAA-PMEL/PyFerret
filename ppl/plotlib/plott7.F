	SUBROUTINE PLOTT7(X,Y,XFACT,YFACT,MPEN,MARK) 
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
C**    @(#)plott7.f	1.1    3/10/88
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
C v6.3 *acm* Fix bug 1668: The thick tics disappear due to pixel aliasing
C            on some long-time-axis plots. Instead use the thick line type
C            for these tics.  
C 12/2022 *acm* Code cleanup: A number of CALL PLOT calls use IPEN which is
C            an array. Instead pass 1st element, ipen(0)
C 
C     THIS ROUTINE DRAWS A LINE FROM THE PEN'S CURRENT 
C     POSITION AND THEN MAKES A MARK.  SEE PLOT ROUTINE IN 
C     COMPLOT DOCUMENTATION.  THIS SUBROUTINE IS THE SAME 
C     AS PLOT EXCEPT THAT SOME OF THE MARKS HAVE BEEN 
C     IMPROVED AND WE NEED TO SPECIFY XFACT AND YFACT. 
C 
C     THE MARKS, MARK=-3,-4,-5,-6,-7,-8 ARE ENLARGEMENTS OF THE 
C     MARKS, MARK=3,4,5,6,7,8. 
C 
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
        INCLUDE 'pen_inc.decl'  ! with num_line_colors
        INCLUDE 'PEN.INC'
        include 'tics_inc.decl'
	include 'TICS.INC'

	integer ithick

	ithick = num_line_colors + 1

	CALL PLOT(X,Y,MPEN,0) 
	CALL VECTRS 
	MMARK=-MARK 
	TY=TLY
	TX=TLX
	GO TO(100,100,100,100,500,600,700,800,900,1100),MMARK 
100	STOP 'MARK' 
C     TO DRAW HORIZONAL TIC OF LENGTH 0.125 AT (X,Y). 
500	TY=TSMY
C     TO DRAW HORIZONAL TIC OF LENGTH 0.25 AT (X,Y). 
600	IF(IY.NE.0)THEN 
	    CALL PLOT(X+IY*TY*.5/XFACT,Y,MPEN,0) 
	ELSE 
	    CALL PLOT(X-TY*.5/XFACT,Y,MPEN,0) 
	    CALL PLOT(X+TY*.5/XFACT,Y,MPEN,0) 
	END IF 
	GOTO 1000
C 
C     TO DRAW VERTICAL TIC OF LENGTH 0.125 AT (X,Y). 
700	TX=TSMX
C     TO DRAW VERTICAL TIC OF LENGTH 0.25 AT (X,Y). 
800	IF(IX.NE.0)THEN 
	    CALL PLOT(X,Y+IX*TX*.5/YFACT,MPEN,0) 
	ELSE 
	    CALL PLOT(X,Y-TX*.5/YFACT,MPEN,0) 
	    CALL PLOT(X,Y+TX*.5/YFACT,MPEN,0) 
	END IF 
	GOTO 1000
C	TO DRAW TRIPLE THICK LARGE X TIC
900	DX=0.005/XFACT
        DX = 0.
	DY=TX*0.5/YFACT
	CALL COLOR(ithick)
	IF(IX.NE.0)THEN
	    CALL PLOT(X,Y+IX*DY,IPEN(0),0)
	    CALL PLOT(X-DX,Y+IX*DY,IPEN(0),0)
	    CALL PLOT(X-DX,Y,IPEN(0),0)
	    CALL PLOT(X+DX,Y,IPEN(0),0)
	    CALL PLOT(X+DX,Y+IX*DY,IPEN(0),0)
	    CALL PLOT(X,Y+IX*DY,IPEN(0),0)
	ELSE
	    CALL PLOT(X,Y-DY,IPEN(0),0)
	    CALL PLOT(X-DX,Y-DY,IPEN(0),0)
	    CALL PLOT(X-DX,Y+DY,IPEN(0),0)
	    CALL PLOT(X+DX,Y+DY,IPEN(0),0)
	    CALL PLOT(X+DX,Y-DY,IPEN(0),0)
	    CALL PLOT(X,Y-DY,IPEN(0),0)
	    CALL PLOT(X,Y+DY,IPEN(0),0)
	ENDIF
	CALL COLOR(1)
	GOTO 1000
C	TO DRAW TRIPLE THICK LARGE Y TIC
1100	DY=0.005/YFACT
        DY = 0.
	DX=TY*0.5/XFACT
	CALL COLOR(ithick)
	IF(IY.NE.0)THEN
	    CALL PLOT(X+IY*DX,Y,IPEN(0),0)
	    CALL PLOT(X+IY*DX,Y-DY,IPEN(0),0)
	    CALL PLOT(X,Y-DY,IPEN(0),0)
	    CALL PLOT(X,Y+DY,IPEN(0),0)
	    CALL PLOT(X+IY*DX,Y+DY,IPEN(0),0)
	    CALL PLOT(X+IY*DX,Y,IPEN(0),0)
	ELSE
	    CALL PLOT(X-DX,Y,IPEN(0),0)
	    CALL PLOT(X-DX,Y-DY,IPEN(0),0)
	    CALL PLOT(X+DX,Y-DY,IPEN(0),0)
	    CALL PLOT(X+DX,Y+DY,IPEN(0),0)
	    CALL PLOT(X-DX,Y+DY,IPEN(0),0)
	    CALL PLOT(X-DX,Y,IPEN(0),0)
	    CALL PLOT(X+DX,Y,IPEN(0),0)
	ENDIF
	CALL COLOR(1)
1000	CALL PLOT(X,Y,IPEN(0),0) 
	RETURN 
	END 
