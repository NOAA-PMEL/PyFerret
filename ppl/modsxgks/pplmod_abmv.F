      SUBROUTINE ABMV
C** 
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
C	 GKSMV  - GKS DRIVER
C
C
#ifdef unix
      include 'pplv11inc/PLTCOM.DAT'
#else
      include 'pplv11inc:PLTCOM.DAT'
#endif
C
#ifdef unix
#ifndef FORTRAN_90
C	FORTRAN 90 requires COMMON block variables to be initialized
C 	in BLOCK DATA *jd* 12.96 for linux

        include 'pplv11inc/PLTDAT_UNIX.DAT'
#endif
#else
       include 'pplv11inc:PLTDAT_VMS.DAT'
#endif

#ifdef FORTRAN_90
C	BLOCK DATA block for linux
	external pltdat_block_data
#endif

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
20	CALL GKSMV
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
