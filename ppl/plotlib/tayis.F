	SUBROUTINE TAYIS(IT,TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,
     *	YF,XHI,XF,TCSIZE,xacross,hgrat,hcolor,hline,color0,is_hours_axis)
C** 
C**    @(#)tayis.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

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
* 	Mod 11.27.95 For 4 digit years J Davison
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C *acm* 03/04  new arguments for plotting graticule lines across plot. 
C V695  *acm*  5/15 Restore color after a graticule using defined color 0
C V695  *acm*  9/15 Changes for hour:min labeling on time axes
C V697  *acm* 12/15 For short-time plots do the time axis in a separate
C                   routine tayis4
* *acm* - 8/16   ticket 1787: restore pen 0 correctly after graticule
* 12/2022 *acm* Code cleanup: Rewrite to replace the computed GOTO statement 
*                             "GOTO (10,20,20,30)IT+1"  with IF logic

C If graticule lines are to be drawn across the plot at the tics
C then only do them for one horizontal and one vertical axis.  
C The dash lines interfere with each other if drawn from both 
C sides. HLINE and VLINE control this.

	INTEGER is_hours_axis
        REAL xacross           ! to draw tic line across plot
        REAL hgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        INTEGER hcolor(*)      ! pen number for graticule lines
	INTEGER color0	       ! pen number to restore 
        LOGICAL hline          ! If true, do the lines across.

	CHARACTER*(*) IT1,IT2

	IF (IT .EQ. 0) THEN

	   IF (is_hours_axis .EQ. 0) THEN
	      CALL TAYIS0(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,YF,XHI,XF,TCSIZE,
     .			  xacross,hgrat,hcolor,hline,color0)
	   ELSE
	      CALL TAYIS4(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,YF,XHI,XF,TCSIZE,
     .			  xacross,hgrat,hcolor,hline,color0,is_hours_axis)
	   ENDIF

	ELSE IF (IT.EQ.1 .OR. IT.EQ.2) THEN

	   CALL TAYIS1(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,YF,XHI,XF,TCSIZE,
     .		       xacross,hgrat,hcolor,hline,color0)


	ELSE IF (IT .EQ. 3) THEN

	   CALL TAYIS3(TLO,IT1,IT2,DT,NMTCT,LTT,IFDB,ISDB,YF,XHI,XF,TCSIZE,
     .		       xacross,hgrat,hcolor,hline,color0)

     	ENDIF

	RETURN
	END
