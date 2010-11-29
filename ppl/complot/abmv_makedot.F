      SUBROUTINE ABMV_makedot
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
C**    @(#)abmv.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C     Mod *JD* 4.5.91 to include PLTDAT.DAT in unix/vms system dependent way
*  
*  8/2001  acm  for PLOT/SYMBOL=DOT
*  Version of ABMV to be called by make_dot.  Call GPM for polymarker
*  rather than GPL (polyline) 

C     CALL ABMV
C
C     ALL PLOTTING EVENTUALLY MUST BE ACCOMPLISHED BY MOVING THE
C     PEN FROM ONE LOCATION TO ANOTHER.  ABMV PERFORMS THIS FUNC-
C     TION BY CALLING THE DRIVER ROUTINES FOR THE VARIOUS PLOTTING
C     DEVICES.  THESE SUBROUTINES ARE LISTED BELOW:
C        TABMV  - TEKTRONIX GRAPHICS TERMINAL
C        PRABMV - PRINTRONIX PRINTER PLOTTER
C        ZABMV  - ZETA PLOTTER
C        BABMV  - BENSON PLOTTER
C	 GKSMV_makedot  - GKS DRIVER calling polymarker
C
C
C *kob* 9/98 - use external statemen to initialize block data rather than an
C              included .DAT file.  the latter was causing mutliple 
C              references 

#ifdef unix
      include 'PLTCOM.DAT'
      external pltdat_block_data
#else
      include 'pplv11inc:PLTCOM.DAT'
      include 'pplv11inc:PLTDAT_VMS.DAT'
#endif
C

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C
C     CHECK THE PLOT TYPE.
C
      GO TO(10,40,30,10,10,20,20),PTYPE + 3
C
C     PLOT ON THE TEKTERMINAL.
C
   10 CALL TABMV
	GOTO 25
20	CALL GKSMV_makedot
C
C     CHECK FOR OTHER PLOTTING DEVICES.
C
25      GOTO(40,50,50,50,30,50,30),PTYPE +3
C
C     PLOT ON THE PRINTRONIX PRINTER PLOTTER.
C
C
C     PLOT ON THE ZETA PLOTTER.
C
   30 CALL ZABMV
	GOTO 50
C
C	PLOT ON THE HP7475 PLOTTER
C
C
40	CALL HPABMV
   50 LPEN = PEN
      RETURN
C
      END
