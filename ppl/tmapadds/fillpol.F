	SUBROUTINE FILLPOL(X,NXS,XF,YF)
C** 
C**    @(#)fillpol.F	1.0    11/30/97
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
C  ARRAY X HOLDS X AND Y'S FOR ALL LINES.  DIMENSIONED FOR NXS.
C  ORDER AS FOLLOWS:  ALL X'S FOR LINE1, ALL X'S FOR LINE2,...
C                     STARTING AT X(NXS/2+1), ALL Y'S FOR LINE1,
C                     ALL Y'S FOR LINE2...
C  XF AND YF ARE FACTORS TO CONVERT TO INCHES FROM USER UNITS
C
*	*sh* 11/97 based on SHADESUB and PLOTXY

#ifdef unix
        include         'tmap_pplv11inc/xinterrupt.cmn' ! 1/89 *sh*
        include         'tmap_pplv11inc/ppl_in_ferret.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif

	DIMENSION X(NXS)

#ifdef unix
        include 'tmap_pplv11inc/parampl5_dat.decl'
        include 'pplv11inc/PARAMPL5.DAT'
        include 'tmap_pplv11inc/axis_inc.decl'
        include 'pplv11inc/AXIS.INC'
        include 'tmap_pplv11inc/lines_inc.decl'
        include 'pplv11inc/LINES.INC'
        include 'tmap_pplv11inc/pen_inc.decl'
        include 'pplv11inc/PEN.INC'
        include 'tmap_pplv11inc/pltl_inc.decl'
        include 'pplv11inc/PLTL.INC'
        include 'tmap_pplv11inc/miss_inc.decl'
        include 'pplv11inc/MISS.INC'
        include 'tmap_pplv11inc/hd_inc.decl'
        include 'pplv11inc/HD.INC'
        include 'tmap_pplv11inc/taxis_inc.decl'
        include 'pplv11inc/TAXIS.INC'
        include 'tmap_pplv11inc/axisl_inc.decl'
        include 'pplv11inc/AXISL.INC'
        include 'tmap_pplv11inc/cmrd_inc.decl'
        include 'pplv11inc/CMRD.INC'
        include 'tmap_pplv11inc/gkscm1_inc.decl'	! sh 11/97
        include 'pplv11inc/GKSCM1.INC'		! sh 11/97
!        include 'tmap_pplv11inc/gkscm2.cmn' ! jd 3.22.89
        include 'tmap_pplv11inc/gkspar.inc'	! sh 11/97
        include 'tmap_pplv11inc/shade_vars.cmn'	! sh 11/97
#else
        INCLUDE 'tmap_pplv11inc:tmap_PARAMPL5.DAT'
        INCLUDE 'tmap_pplv11inc:tmap_AXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_LINES.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PEN.INC'
        INCLUDE 'tmap_pplv11inc:tmap_PLTL.INC'
        INCLUDE 'tmap_pplv11inc:tmap_HD.INC'
        INCLUDE 'tmap_pplv11inc:tmap_MISS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_TAXIS.INC'
        INCLUDE 'tmap_pplv11inc:tmap_AXISL.INC'
        INCLUDE 'tmap_pplv11inc:tmap_CMRD.INC'
#endif

	INTEGER xpolybase, ypolybase, zpolybase, polysizemax, nl,
     .		nvertex, fill_ndx			! 11/97

	INTEGER GET_NDX

* GKS initialization vars
* v491 *kob* removed unnecessary variable declarations
	integer		error,trans_no,cflag
        integer         my_trans
        data            my_trans/20/
	real	 	w(4),v(4),ndc(4),uc(4)
	real		xinv,yinv

	logical		valid

*************************************
* INITIALIZE
	nl = lnum - 1	! # lines over which to loop: last line is color info

* determine temp memory buffer location for points translated to plot coords
* ... layout is xline1, xline2, ..., xlinenl, xcolor , bufferspace ...
*               yline1, yline2, ..., ylinenl, ycolor , bufferspace ...
*     "zpolybase" is the offset to the ycolor array
	xpolybase = 1
	DO 2010  I=1,nl
 2010	xpolybase=xpolybase+LLENG(I)
	zpolybase = xpolybase + nxs/2
	xpolybase = xpolybase + lleng(lnum)	! skip past color array
	ypolybase = xpolybase + nxs/2
	polysizemax = nxs - ypolybase	! unused buffer space
	fill_ndx = 0

* set up GKS environment (code copied from SHADE_SUB)
*	DETERMINE PATTERNS OR COLORS TO SET BUNDLED FILL AREA REPRESENTATIONS
	call set_fill_area_rep (wsid,wstype)

*	GET PROPER SCALING, NDC COORDINATES, ETC FOR FILL
	call gqcntn (error,trans_no)
	if (error .ne. 0) goto 1100

	call gqnt (trans_no,error,w,v)
	if (error .ne. 0) goto 1100

	call get_user_coords (w,v,ndc,uc,xinv,yinv)

*	NOW SET THE CLIPPING RECTANGLE AROUND THEM AND PATTERN SIZE
* v4.91 *kob* had to add <cr> after first line below
	call gswn (my_trans,uc(1)*xinv,uc(2)*xinv,uc(3)*yinv,uc(4)*yinv)
        call gsvp (my_trans,ndc(1),ndc(2),ndc(3),ndc(4))
	call gselnt (my_trans)

	call gqclip (error,cflag,ndc)
	if (error .ne. 0) goto 1000
	if (cflag .ne. gclip) call gsclip (gclip)

*************************************

* original PLOT+ code 
	IBASE=0
	IY=NXS/2
	ZT=YF/XF
	ICNT=1
	IF(M.GE.2.AND.IFLG(2).NE.0)ICNT=P(2)

******** LOOP OVER LINES STORED
	DO 200 I=1,nl
	  CALL MARKH(HMARK(I))
	  CALL COLOR(IPEN(I))

* will this polygon fit in the buffer?
	  IF (lleng(i) .GT. polysizemax) THEN
	     WRITE (6,*) 'Too much polygon data to plot at once'
	     GOTO 200
	  ENDIF 
* determine the color index for the fill
	   fill_ndx = GET_NDX (x(zpolybase),fill_ndx)
	   if (fill_ndx .eq. -999) goto 200

* assemble the vertices into arrays in units of plot coordinates
* note: all points in the polygon are guaranteed to be "valid"
	   nvertex = 0
	   DO 100 J=IBASE+1,IBASE+LLENG(I)
	      CALL TRANS(I,X(J),X(IY+J),XT,YT)
	      nvertex = nvertex + 1
	      x(xpolybase+nvertex-1) = xt
	      x(ypolybase+nvertex-1) = yt
 100	   CONTINUE

* draw the filled polygon
	   CALL SET_FILL_NDX (fill_ndx)
	   CALL GFA(nvertex, x(xpolybase), x(ypolybase)) 

	   if (interrupted) go to 1000 !*jd* 8.3.92
	   IBASE=IBASE+LLENG(I)		! prepare for next line
 200	CONTINUE

******** END OF LOOP OVER LINES STORED

* ... restore GKS transforms following FILLPOL 
 1000	call gsclip (cflag)
	call gselnt (trans_no)

* ... and DO THE KEY
	if (do_key .AND. .NOT.interrupted) call key

 1100	CALL COLOR(IPEN(0))

	RETURN
	END



