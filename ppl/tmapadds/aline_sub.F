	SUBROUTINE ALINE_SUB (x,nxs,a_line,axmin,aymin,axmax,aymax)

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
*	Modified PPLMOD_PLTIT and PPLMOD_PLOTXY to execute the ALINE command 
*	where a line is drawn of a particular type as specified.
*	J Davison 10.4.89

*****************************************************
* local variable declarations for PLTIT (for TMAP)
	REAL  	x
	INTEGER nxs
* TMAP include files with explicit type declarations:
#ifdef unix
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'
#else
	INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
	INCLUDE 'PPLV11INC:PARAMPL5.DAT'
#endif


CCCCC	VIRTUAL X(NXS)		(*jd* 12.96 for linux port) 
	dimension X(NXS)     

	real 		cs,sn,xt,yt,oldx(4)
	real		axmin,aymin,axmax,aymax
	integer  	icnt,im,j,ibase,iy,a_line
	logical		old_swait


C	KTYPE = 1 -XY  0 -Z  2-VECTOR
#ifdef unix
	include 'lines_inc.decl'
	include 'LINES.INC'
	include 'pen_inc.decl'
	include 'PEN.INC'
	include 'axis_inc.decl'
	include 'AXIS.INC'
	include 'hd_inc.decl'
	include 'HD.INC'
	include 'taxis_inc.decl'
	include 'TAXIS.INC'
	include 'axisl_inc.decl'
	include 'AXISL.INC'
	include 'plt_inc.decl'
	include 'PLT.INC'
	include 'pltl_inc.decl'
	include 'PLTL.INC'
	include 'mplot_inc.decl'
	include 'MPLOT.INC'
	include 'cmrd_inc.decl'
	include 'CMRD.INC'
	include 'prmcom_inc.decl'
	include 'PRMCOM.INC'
	include 'miss_inc.decl'
	include 'MISS.INC'
	include 'switch_inc.decl'
	include 'SWITCH.INC'
	include 'system_inc.decl'
	include 'SYSTEM.INC'
	include 'cont_inc.decl'  !added to get ivcomp v1.1
	include 'CONT.INC'  !added to get ivcomp v1.1
        include 'gkscm2.cmn' ! jd 3.22.89
#else
	INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
	INCLUDE 'PPLV11INC:LINES.INC'
	INCLUDE 'TMAP_PPLV11INC:PEN_INC.DECL'
	INCLUDE 'PPLV11INC:PEN.INC'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:HD_INC.DECL'
	INCLUDE 'PPLV11INC:HD.INC'
	INCLUDE 'TMAP_PPLV11INC:TAXIS_INC.DECL'
	INCLUDE 'PPLV11INC:TAXIS.INC'
	INCLUDE 'TMAP_PPLV11INC:AXISL_INC.DECL'
	INCLUDE 'PPLV11INC:AXISL.INC'
	INCLUDE 'TMAP_PPLV11INC:PLT_INC.DECL'
	INCLUDE 'PPLV11INC:PLT.INC'
	INCLUDE 'TMAP_PPLV11INC:PLTL_INC.DECL'
	INCLUDE 'PPLV11INC:PLTL.INC'
	INCLUDE 'TMAP_PPLV11INC:MPLOT_INC.DECL'
	INCLUDE 'PPLV11INC:MPLOT.INC'
	INCLUDE 'TMAP_PPLV11INC:CMRD_INC.DECL'
	INCLUDE 'PPLV11INC:CMRD.INC'
	INCLUDE 'TMAP_PPLV11INC:PRMCOM_INC.DECL'
	INCLUDE 'PPLV11INC:PRMCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:MISS_INC.DECL'
	INCLUDE 'PPLV11INC:MISS.INC'
	INCLUDE 'TMAP_PPLV11INC:SWITCH_INC.DECL'
	INCLUDE 'PPLV11INC:SWITCH.INC'
	INCLUDE 'TMAP_PPLV11INC:SYSTEM_INC.DECL'
	INCLUDE 'PPLV11INC:SYSTEM.INC'
	INCLUDE 'TMAP_PPLV11INC:CONT_INC.DECL'  !Added to get IVCOMP v1.1
	INCLUDE 'PPLV11INC:CONT.INC'  !Added to get IVCOMP v1.1
        INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc' ! JD 3.22.89
#endif

	CALL COLOR(IPEN(0))
!	CALL SETAX(X,NXS,XF,YF,XL,YL,XH,YH,NMX,NMY,XTI,XTL)

	oldx(1) 	= x(1)
	x(1) 		= axmin
	oldx(2) 	= x(2)
	x(2) 		= axmax

	oldx(3)		= x(nxs/2+1)
	x(nxs/2+1) 	= aymin
	oldx(4)		= x(nxs/2+2)
        x(nxs/2+2) 	= aymax

	IBASE=0
	IY=NXS/2
	CS=COS(VANG*.01745329252)
	SN=SIN(VANG*.01745329252)
!	ZT=YF/XF
	ICNT=1

	old_swait = swait
	swait = .false.
	CALL MARKH(HMARK(a_line))
	CALL COLOR(IPEN(a_line))

	IF(ITYPE(a_line).GE.4 .and. .not.gks_open)THEN ! JD 3.22.89 NEW PPL
!	IF(ITYPE(a_line).GE.4) THEN ! OLD PPL (PRE JD 89 GKS)
	   CALL DASHSZ(DN1(a_line),UP1(a_line),DN2(a_line),UP2(a_line))
	   CALL DASHES
	ELSE
	   CALL VECTRS
	END IF

	IF(ITYPE(a_line).EQ.3)CALL POINTS

	CALL TRANS(1,X(IBASE+1),X(IY+IBASE+1),XT,YT)                
	CALL PLOT(XT,YT,0,0)
	IF (ITYPE(a_line).NE.0.AND.ITYPE(a_line).NE.4) 
     .	CALL MARK(IMARK(a_line))
	IM=0
	IF (ITYPE(a_line).EQ.1.OR.ITYPE(a_line).EQ.3)IM=IMARK(a_line)
	DO 60 J=1,2
	   CALL TRANS(1,X(J),X(IY+J),XT,YT)
	   CALL PLOT(XT,YT,1,0)
60      CALL MARK(IM)
	IF (ITYPE(a_line).EQ.2.OR.ITYPE(a_line).EQ.5) 
     .	CALL MARK(IMARK(a_line))
 	CALL COLOR(IPEN(0))

	x(1) 		= oldx(1)
	x(2) 		= oldx(2)

	x(nxs/2+1) 	= oldx(3)
        x(nxs/2+2) 	= oldx(4)

	swait = old_swait

!	CALL SCALE(XF,YF,XORG,YORG,XLO,YLO)
C
	CALL WINDOW(0.,0.,0.,0.)
C
!	IF(ITFLG.EQ.1)THEN
!	    XHI=XTI
!	    XLO=XTL
!	ELSE IF(ITFLG.EQ.-1)THEN
!	    YHI=XTI
!	    YLO=XTL
!	ENDIF
	RETURN
	END
