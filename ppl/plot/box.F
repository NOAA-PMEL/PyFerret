      SUBROUTINE BOX 
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
