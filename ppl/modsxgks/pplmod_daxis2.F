	SUBROUTINE DAXIS2(XF,YF,NMX,NMY)
C** 
C**    @(#)daxis2.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
*     MOD *JD* 10.91 to eliminate PLOT COMPLETE msg if in ferret

#ifdef unix
	INCLUDE 'pplv11inc/PARAMPL5.DAT'
	INCLUDE 'pplv11inc/AXIS.INC'
	INCLUDE 'pplv11inc/LABELS.INC'
	INCLUDE 'pplv11inc/LABCOM.INC'
	INCLUDE 'pplv11inc/PLT.INC'
	INCLUDE 'pplv11inc/HAB.INC'
	INCLUDE 'pplv11inc/LINES.INC'
	INCLUDE 'pplv11inc/MPLOT.INC'
	INCLUDE 'pplv11inc/TAXIS.INC'
	INCLUDE 'pplv11inc/TICS.INC'
	INCLUDE 'pplv11inc/TICS2.INC'
	INCLUDE 'pplv11inc/AXISL.INC'
	INCLUDE 'pplv11inc/XYLABP.INC'
	INCLUDE 'pplv11inc/CMRDL.INC'
	INCLUDE 'pplv11inc/LUNITS.INC'
	INCLUDE 'pplv11inc/PRMCOM.INC'
	INCLUDE 'pplv11inc/SYSTEM.INC'
	INCLUDE 'pplv11inc/SWITCH.INC'
#else
        INCLUDE 'pplv11inc:PARAMPL5.DAT'
        INCLUDE 'pplv11inc:AXIS.INC'
        INCLUDE 'pplv11inc:LABELS.INC'
        INCLUDE 'pplv11inc:LABCOM.INC'
        INCLUDE 'pplv11inc:PLT.INC'
        INCLUDE 'pplv11inc:HAB.INC'
        INCLUDE 'pplv11inc:LINES.INC'
        INCLUDE 'pplv11inc:MPLOT.INC'
        INCLUDE 'pplv11inc:TAXIS.INC'
        INCLUDE 'pplv11inc:TICS.INC'
        INCLUDE 'pplv11inc:TICS2.INC'
        INCLUDE 'pplv11inc:AXISL.INC'
        INCLUDE 'pplv11inc:XYLABP.INC'
        INCLUDE 'pplv11inc:CMRDL.INC'
        INCLUDE 'pplv11inc:LUNITS.INC'
        INCLUDE 'pplv11inc:PRMCOM.INC'
        INCLUDE 'pplv11inc:SYSTEM.INC'
        INCLUDE 'pplv11inc:SWITCH.INC'
	byte inc
#endif

#ifdef unix
        include         'tmap_pplv11inc/xinterrupt.cmn' ! 1/89 *sh*
        include         'tmap_pplv11inc/ppl_in_ferret.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif 

******************************************************************************

	CHARACTER OUT*255
	CALL VECTRS
	DO 80 I=1,NLAB
	IF(RHLABS(I).LT.0.0)THEN
		HLB=HLABS
	ELSE
		HLB=RHLABS(I)
	ENDIF
	CS=COS(RLABS(I)*0.0175329252)
	SN=SIN(RLABS(I)*0.0175329252)
	IAT=0
	SHFT=0.0
	NCR=LNBLK(LABS(I),255)
	XMN=SYMWID(HLB,NCR,LABS(I))
	IF(USRLBS(I))THEN
	    XFUSER=1.
	    YFUSER=1.
	    YLOUSR=0.
	    XLOUSR=0.
	ELSE
	    XFUSER=XF
	    YFUSER=YF
	    XLOUSR=XLO
	    YLOUSR=YLO
	ENDIF
	IF(XLABS(I).LT.1.E35)THEN
	    XL=XLABS(I)
	    YL=YLABS(I)
	    LTY=LTYPE(I)
	    IF(LTY.GT.0)THEN
		SHFT=-XMN
	    ELSE IF(LTY.EQ.0)THEN
		SHFT=-XMN*.5
	    ENDIF
	    CALL ARROW(XL/XFUSER+XLOUSR,YL/YFUSER+YLOUSR,
     *	    XLLABS(I)/XFUSER+XLOUSR,YLLABS(I)/YFUSER+YLOUSR,
     *	    LLTYPE(I),SHFT,XMN,CS,SN,XF,YF,HLB)
	    GOTO 70
	ENDIF
	ICHAR=IGRNPT(XL,YL)
	LTY=-1
	IF(ICHAR.EQ.67.OR.ICHAR.EQ.99)THEN
	    LTY=0
	    SHFT=-XMN*.5
	ELSE IF(ICHAR.EQ.82.OR.ICHAR.EQ.114)THEN
	    LTY=1
	    SHFT=-XMN
	ELSE IF(ICHAR.EQ.76.OR.ICHAR.EQ.70.OR.ICHAR.EQ.108
     *	.OR.ICHAR.EQ.102)THEN
	    IAT=1
	    XL2=XL
	    YL2=YL
	    ICHARO=ICHAR
	    ICHAR=IGRNPT(XL,YL)
	    IF(ICHAR.EQ.67.OR.ICHAR.EQ.99)THEN
		SHFT=-XMN*.5
		LTY=0
	    ENDIF
	    IF(ICHAR.EQ.82.OR.ICHAR.EQ.114)THEN
		SHFT=-XMN
		LTY=1
	    ENDIF
	    IF(ICHARO.EQ.70.OR.ICHARO.EQ.102)IAT=2
	    CALL ARROW(XL,YL,XL2,YL2,IAT,SHFT,XMN,CS,SN,XF,YF,HLB)
	    WRITE(OUT,998)I,(XL2-XLOUSR)*XFUSER,(YL2-YLOUSR)*YFUSER,IAT
998 	    FORMAT('C LLABS,',I2,2(',',1PE10.3),',',I2)
	    CALL ECHO(OUT,40)
	ENDIF
	XL=(XL-XLOUSR)*XFUSER
	YL=(YL-YLOUSR)*YFUSER
	IF(USRLBS(I))THEN
	    WRITE(OUT,999)I,XL,YL,LTY,LABS(I)(:NCR)
999	    FORMAT('C LABS,',I2,2(',',1PE10.3),',',I2,',','"',A,'"')
	ELSE
	    WRITE(OUT,997)I,XL,YL,LTY,LABS(I)(:NCR)
997	    FORMAT('C LABS/NOUSER,',I2,2(',',1PE10.3),',',I2,
     *	    ',','"',A,'"')
	ENDIF
	NOUT=LNBLK(OUT,255)
	CALL ECHO(OUT,NOUT)
70	XL=XL/XFUSER+XLOUSR+SHFT*CS/XF
	YL=YL/YFUSER+YLOUSR+SHFT*SN/YF
	CALL SYMBEL(XL,YL,RLABS(I),HLB,NCR,LABS(I))
80	CONTINUE
	IF(PLTOPN)RETURN
	IF(IMULT.EQ.0.OR.IMCNT.EQ.MNX*MNY)THEN
	    IF(SWAIT)THEN
		CALL HABIS(IPLOT,IPAUSE,IHDCPY,INC)
	    ELSE
		CALL HABIS(IPLOT,0,IHDCPY,INC)
	    ENDIF

*     Mod is here
        if (ppl_in_ferret) then   
           call ferret_plot_complete (icount)
        else
	   IF(.NOT.QUIETF)WRITE(LTTOUT,'('' PLOT'',I8,
     *         '' COMPLETE'')')ICOUNT
	       IF(INC.EQ.27)THEN
		  CALL ATFLSH
	       ENDIF
	endif

        ENDIF
	RETURN
	END

