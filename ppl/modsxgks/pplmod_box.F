      SUBROUTINE BOX 
C** 
C**    @(#)box.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

*     11.6.92 *jd* Mod to use viewport sizes when in FERRET

#ifdef unix
        include 'tmap_pplv11inc/parampl5_dat.decl'
	INCLUDE 'pplv11inc/PARAMPL5.DAT'
        include 'tmap_pplv11inc/plt_inc.decl'
	INCLUDE 'pplv11inc/PLT.INC'
        include 'tmap_pplv11inc/pen_inc.decl'
	INCLUDE 'pplv11inc/PEN.INC'
#else
        INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
	INCLUDE 'PPLV11INC:PARAMPL5.DAT'
        INCLUDE 'TMAP_PPLV11INC:PLT_INC.DECL'
	INCLUDE 'PPLV11INC:PLT.INC'
        INCLUDE 'TMAP_PPLV11INC:PEN_INC.DECL'
	INCLUDE 'PPLV11INC:PEN.INC'
#endif

#ifdef unix
        include 'tmap_pplv11inc/ppl_in_ferret.cmn'
#else
        INCLUDE 'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif

        real     vwidth,vheight,width2,height2

C 	stmt added for linux *jd* 12.96
	external scale

*******************************************************************************

        if (ppl_in_ferret) then
           call get_view_size (vwidth,vheight)
           width2 = vwidth
           height2 = vheight
        else
           width2 = width
           height2 = height
        endif

	CALL SCALE(1.,1.,0.,0.,0.,0.) 
	IF(IP(1).NE.1)RETURN
	CALL PLOT(.01,.01,0,0) 
	CALL PLOT(WIDTH2-.02,.01,1,0) 
	CALL PLOT(WIDTH2-.02,HEIGHT2-.04,1,0) 
	CALL PLOT(.01,HEIGHT2-.04,1,0) 
	CALL PLOT(.01,.01,1,0) 
	RETURN 
	END 
