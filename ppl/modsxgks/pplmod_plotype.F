      SUBROUTINE PLTYPE(ICODE)
C**
C**    @(#)plotype.f	1.1    3/10/88
*
*	Modified 3.15.89 J Davison JISAO/PMEL/TMAP
*	For GKS plots (pltype = 3 or 4) captures decision to write to
*	GKS metafile, and sets up GKS as required.
*
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
C     PLTTER DEPENDENT ROUTINE
C     THIS ROUTINE MUST BE MODIFIED EVERY TIME NEW PLT DEVICES ARE ADDE
C     OR OLD ONES ARE DELETED.
C
C
C     CALL PLTYPE(ICODE)
C
C     THIS SUBROUTINE SELECTS THE DEVICE OR DEVICES THAT WILL BE
C     USED FOR PLTTING.
C
C     THE VARIOUS COMBINATIONS OF PLTTING DEVICES THAT CAN BE
C     DRIVEN BY COMPLT AND THE ASSOCIATED PLT TYPES ARE LISTED
C     BELOW:
C	 GKS AND ZETA PLOTTER               -    4
C        GKS ONLY                           -    3
C        TEKTERMINAL AND ZETA PLOTTER       -    2
C        TEKTERMINAL ONLY                   -    1
C        ZETA PLOTTER ONLY                  -    0
C	 HP7475 PLOTTER ONLY		    -   -1
C	 HP7475 PLOTTER AND TEKTERMINAL     -   -2
C
C
	integer	icode

#ifdef unix
	include 'tmap_pplv11inc/pltcom_dat.decl'
	include 'pplv11inc/PLTCOM.DAT'
	include 'tmap_pplv11inc/cmrdl_inc.decl'
	include 'pplv11inc/CMRDL.INC'
#else
	INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL'
	INCLUDE 'PPLV11INC:PLTCOM.DAT'
	INCLUDE 'TMAP_PPLV11INC:CMRDL_INC.DECL'
	INCLUDE 'PPLV11INC:CMRDL.INC'
#endif

	IF(PLTFLG)THEN
	    PEN=.FALSE.
	    IF(PTYPE.LT.0)THEN
		A=0.
		B=0.
		CALL HPABMV
		CALL ZBUFFT('SP0;',4)
		OLDHTA=0
		CALL HFLUSH
	    ELSE IF(PTYPE.EQ.0.OR.PTYPE.EQ.2.OR.PTYPE.EQ.4)THEN
		CALL ZABMV
		CALL BINFSH
	    ENDIF
	    PLTFLG=.FALSE.
	ENDIF
      PTYPE = ICODE
C
C     CHECK FOR A LEGAL PLT TYPE.
C
      IF(IABS(PTYPE) .GT. 4)PTYPE = 0

*     DETERMINE IF METAFILE WILL BE USED IN GKS, OPEN GKS IF NOT OPEN
	if ((ptype .eq. 3 .or. ptype .eq. 4) .and. icom(1:6) .ne. 'CLSPLT') 
     .		call set_gks_metafile		
C
C
C     REINITIALIZE THE TEKTERMINAL TYPE IN CASE THIS IS NOT THE
C     FIRST TIME THIS SUBROUTINE HAS BEEN CALLED.
C
      CALL TKTYPE(4010)
C
C     INTIALIZE THE PRINTRONIX RASTER BUFFER IF NEEDED.
C
      RETURN
C
      END
