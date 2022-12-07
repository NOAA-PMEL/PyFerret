	SUBROUTINE PLOTXY_RIBBON(X,NXS,XF,YF)
C** 
C**    @(#)plotxy.F	1.1    3/10/88
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
C  Ribbon Plots: lines colored by a variable along the line.
C  8/2012 ACM
C  For a ribbon plot, x's are at the start, Y1 are the Y values, and
C  Y2 are values by which to color the lines.
C
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

*  ACM 10/2012    Modulo-ing for PLOT/VS longitudes overlaid on a map.
* 1/2013  *acm* Fix ticket 2029, call to WS_LINE_SPECTRUM was missing an 
*               argument for setting up metafile plots.
*  ACM  2/2013    ticket 2141. Fix the modulo operation when the longitude
*                 range is over 360.
* ACM   5/2014  Fixes for changes of color in combination with handling
*               modulo longitude data
* V6.92 *acm* 6/14 New color /key options: any of /key=horizontal,centerlabel
* V6.94 *acm* 1/15 fix ticket 2216: when using (-inf), a color-by variable
*                  starting with missing was colored incorrectly.
* V6.94+ *acm* 3/15 Ticket 2247: singleton points in line plots marked with a dot
* v696 *acm*   6/15 PLOT/VS/RIBBON/GAPLOC to define seperate line segments on
*                   color PLOT/VS plots (LAS ticket 1597 and others)
* V698  4/16 *acm* New PLOT/NTH=
* V73  3/18 *kms*  For clarity, append MD to makedot.cmn values IPT, XDOT, 
*                  and YDOT; make MDOT (99) a makedot.cmn parameter DOTSYMIM
* V762  8/20 *acm* Don't draw outside plot edges when doing modulo correction.
* V765  9/22 *acm* Issue 118: Restore mark size to default after possible PLOT/SIZE=
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.
****************** NO? If we do this then symbols in a line-key are wrong size *****

        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'

	INTEGER NXS
	REAL X(*), XS, YS

        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'pen_inc.decl'
        include 'PEN.INC'
        include 'pltl_inc.decl'
        include 'PLTL.INC'
        include 'hd_inc.decl'
        include 'HD.INC'
        include 'miss_inc.decl'
        include 'MISS.INC'
        include 'taxis_inc.decl'
        include 'TAXIS.INC'
        include 'axisl_inc.decl'
        include 'AXISL.INC'
        include 'cmrd_inc.decl'
        include 'CMRD.INC'
	include 'gkscm1_inc.decl'   
	include 'GKSCM1.INC'
        include 'gkscm2.cmn'
        include 'gkspar.inc'
        include 'shade_vars.cmn'
        include 'makedot.cmn'
	include 'cont_inc.decl'
	include 'CONT.INC'

	REAL datum, xlast, ylast, frac, dx, dy, xi, yi, xhalf, yhalf
	INTEGER xpolybase, ypolybase, zpolybase, polysizemax, nl,
     .		nvertex, low_index, colr_ndx, last_colr_ndx, ncolors, 
     .		line_ndx, thick_offset, nseg, iseg, indx, inc, nskp, idec
	INTEGER GET_NDX, ibase, iline2, ilineg

	logical	isbad, last_bad, bad_blank, new_branch, md

        real            xadd, xaddlast, datx, daty, datxlast,datylast
        integer         ndx,type
	
	INTEGER		klast
	REAL		xtlast, ytlast


* GKS initialization vars
	integer		error,trans_no,cflag
        integer         my_trans
        data            my_trans/20/
	real	 	w(4),v(4),ndc(4),uc(4)
	real		xinv,yinv

*************************************

	nl = lnum

	xadd = 0.
	xadlast = 0.
	new_branch = .FALSE.
C
C  JTYPE IS DATA TYPE (0 FOR 2D, 1 FOR 1D)
C
	IF(JTYPE.NE.1) RETURN
	I = 1

	IBASE=0
	IY=NXS/2
	CS=COS(VANG*.01745329252)
	SN=SIN(VANG*.01745329252)
	ZT=YF/XF
	ICNT=1
	IF(M.GE.2.AND.IFLG(2).NE.0)ICNT=P(2)

* Line 2 contains the variable on which to color. ZMAX and ZMIN are its 
* range.  Set levels and set up color line bundles.

C  Set the contour/color levels

	CALL SET_LEVELS_SHADE_FILL (status)
	shd_levels = MAX (nlev2-1,1)  ! as in shade_sub
	
* DETERMINE COLORS TO SET BUNDLED LINE REPRESENTATIONS
	CALL WS_LINE_SPECTRUM (wsid,wstype, low_index)

* For plotting missing values of the color-by variable.
* Code -1 is for the default, thin black line (or gray? Set in ws_line_spectrum)
* Code -2 is to lift the pen and make the segment blank.
* If a palette, then use the first color. This was set up in ws_line_spectrum.

	bad_blank = .FALSE.
	IF (ribbon_miss(1) .EQ. -2) THEN 
	   bad_blank = .TRUE.
	ENDIF

        ILINE2=IBASE+LLENG(I)
        ILINEG=IBASE+2*LLENG(I)
        igap = 0

C set decimation for symbols
        nskp = MAX(1,nskpsym(i))

        CALL MARKH(HMARK(I))

* IPEN contains the thickness setting.

	  ncolors = shd_levels
	  if (rgb_mapping .eq. by_level) ncolors = ncl

	  thick_offset = 0
	  IF (ipen(i) .EQ. 2) thick_offset = (ncolors+1) 
	  IF (ipen(i) .EQ. 3) thick_offset = 2*(ncolors+1) 

C     SET PLOT MODE

* standard line or point plot

	    IF(ITYPE(I).GE.4)THEN ! JD 3.22.89
	      CALL DASHSZ(DN1(I),UP1(I),DN2(I),UP2(I))
	      CALL DASHES
	    ELSE
	      CALL VECTRS
	    END IF
	    IF(ITYPE(I).EQ.3)THEN 
               CALL POINTS
            ENDIF
	    IF(.NOT.(XLE.OR.XEQ.OR.XGE.OR.YLE.OR.YEQ.OR.YGE))THEN
C     DO NOT CHECK POINTS FOR MISSING DATA
	      CALL TRANS(I,X(IBASE+1),X(IY+IBASE+1),XT,YT)
	      CALL PLOT(XT,YT,0,0)
	      IF(ITYPE(I).NE.0.AND.ITYPE(I).NE.4)CALL MARK(IMARK(I))
	      IM=0
	      IF(ITYPE(I).EQ.1.OR.ITYPE(I).EQ.3)IM=IMARK(I)
	      DO 60 J=IBASE+1,IBASE+LLENG(I)
	        CALL TRANS(I,X(J),X(IY+J),XT,YT)
	        CALL PLOT(XT,YT,1,0)
	        CALL MARK(IM)
60	      CONTINUE
	    ELSE
C     CHECK POINTS FOR OUT OF RANGE   (CM)
	      DO 70 J=IBASE+1,IBASE+LLENG(I)
* ... locate first point
		IF(XLE.AND.X(J).LE.CMXLE)GOTO 70
		IF(XEQ.AND.X(J).EQ.CMXEQ)GOTO 70
		IF(XGE.AND.X(J).GE.CMXGE)GOTO 70
		IF(YLE.AND.X(IY+J).LE.CMYLE)GOTO 70
		IF(YEQ.AND.X(IY+J).EQ.CMYEQ)GOTO 70
		IF(YGE.AND.X(IY+J).GE.CMYGE)GOTO 70
		
		last_bad = .FALSE.
		
		datum = X(IY+J+ILINE2)
		IF (datum .EQ. -3.E33) datum = CMZGE  ! This value set in pplldx.
		  
	        colr_ndx = GET_NDX (datum,colr_ndx)
		CALL SET_RIBBON_NDX (colr_ndx)

	        line_ndx = colr_ndx + thick_offset + low_index - 1
	        IF (colr_ndx .EQ. -999) THEN 
		   colr_ndx = bad_ribbon_ndx
		   line_ndx = colr_ndx + low_index - 1
		   last_bad = .TRUE.
		ELSE
		   nothing = 0
		ENDIF

		CALL COLOR (line_ndx) 
		last_colr_ndx = colr_ndx
	
	        datx = X(J)
	        daty = X(IY+J)
		IF (MOD_VS_X .AND. datx .LT. xlo) THEN
		   xadd = 360.
		   datx = datx + xadd
		ELSE IF (MOD_VS_X .AND. datx .GT. xhi) THEN
		   xadd = -360.
		   datx = datx + xadd
		ENDIF
		xaddlast = xadd

	        CALL TRANS(I,datx,daty,XT,YT)
	        CALL PLOT(XT,YT,0,0)
		xlast = xt
		ylast = yt
		isbad = .FALSE.
	        IF(ITYPE(I).NE.0.AND.ITYPE(I).NE.4)CALL MARK(IMARK(I))
 	        IM=0
	        IF(ITYPE(I).EQ.1.OR.ITYPE(I).EQ.3)IM=IMARK(I)
	        GOTO 71
70	      CONTINUE
71	      IDN=1
	      idec =  0
	      IF (last_bad .AND. bad_blank) IDN = 0
* ... plot remaining points
	      DO 72 K=J+1,IBASE+LLENG(I)
		IF((XLE.AND.X(K).LE.CMXLE).OR.
     *		   (XEQ.AND.X(K).EQ.CMXEQ).OR.
     *		   (XGE.AND.X(K).GE.CMXGE).OR.
     *		 (YLE.AND.X(IY+K).LE.CMYLE).OR.
     *		 (YEQ.AND.X(IY+K).EQ.CMYEQ).OR.
     *		 (YGE.AND.X(IY+K).GE.CMYGE))THEN

		  IF (IDN .EQ. 1 .AND. klast.EQ.k-1) THEN
*                     prev pt was a singleton
*                     plot a dot for line plots without symbols
                      IF(ITYPE(I).EQ.0.OR.ITYPE(I).EQ.4) THEN
                          CALL PLOT(XTLAST,YTLAST,0,0)
                          CALL MARK(DOTSYMIM)
                      ENDIF
		  ENDIF

	          IDN=0
	        ELSE
		
	          CALL PLOT(XT,YT,IDN,0)

	          datx = X(K)
	          daty = X(IY+K)

* if 1, then this is the end of a segment. Draw to it, and pick up the pen.
		  IF (gap_var.GT.0) igap = INT(X(IY+K+ILINEG)) 

		  IF (MOD_VS_X .AND. datx .LT. xlo) THEN 
		     xadd = 360.
		     datx = datx + xadd

		  ELSE IF (MOD_VS_X .AND. datx .GT. xhi) THEN  
		     xadd = -360.
		     datx = datx + xadd

		  ELSE
		     xadd = 0.

		  ENDIF

		  IF (xadd .NE. xaddlast) THEN
		     datxlast = datx - xadd + xaddlast
		     IF (datxlast.LT.xhi .AND. datxlast.GT.xlo) THEN
		        CALL TRANS(I,datxlast,daty,XT,YT)
		        CALL PLOT(xt,yt,idn,0)
		     ENDIF
		     CALL TRANS(I,datx,daty,XT,YT)
		     CALL PLOT(XT,YT,0,0)
		     CALL PLOT(XT,YT,IDN,0)
		     new_branch = .TRUE.
		  ENDIF

	          CALL TRANS(I,datx,daty,XT,YT)
		  xaddlast = xadd

! in LINE/FAST mode go halfway between the last point and this one, using previous color.
	          IF (ITYPE(I).NE.3) THEN
		     xhalf = xlast + (xt - xlast)/2
		     yhalf = ylast + (yt - ylast)/2

		     IF ((.NOT.new_branch)) THEN 
		     IF (RIBBON_FAST .OR. last_bad) THEN 
                        IF (last_bad .AND. bad_blank) THEN 
		           CALL PLOT(xhalf,yhalf,0,0)
		        ELSE
		           CALL PLOT(xhalf,yhalf,IDN,0)
		        ENDIF
		     ENDIF
		     ENDIF
		  ENDIF

		  datum = X(IY+K+ILINE2)
		  IF (datum .EQ. -3.E33) datum = CMZGE  ! This value set in pplldx.
		  
	          colr_ndx = GET_NDX (datum,colr_ndx)
		  CALL SET_RIBBON_NDX (colr_ndx)
	          line_ndx = colr_ndx + thick_offset + low_index - 1

	          IF (colr_ndx .EQ. -999 .AND.(.NOT.new_branch)) THEN
		     isbad = .true.
		     colr_ndx = bad_ribbon_ndx
		     line_ndx = colr_ndx + low_index - 1

		     IF ((bad_blank .AND. .NOT.last_bad) .AND. 
     .                   (ITYPE(I).NE.3)) CALL PLOT(xhalf,yhalf,IDN,0)
		     IF (bad_blank) IDN = 0
		ELSE
		   nothing = 0
		ENDIF
		  
	          IF (line_ndx .LT. 1) line_ndx = 1

	          IF ( colr_ndx.NE.last_colr_ndx ) THEN
		     
* In default (not fast) mode, draw sub-segments filling in all the colors
* corresponding to the index range between the color at this point and the
* color at the previous point.

		     IF (.NOT.RIBBON_FAST .AND. 
     .                    .NOT.isbad .AND. .NOT.last_bad .AND.
     .                    last_colr_ndx.GT.0 .AND.
     .                    ABS(colr_ndx-last_colr_ndx) .GT. 1 .AND.
     .                    (.NOT.new_branch) .AND. ITYPE(I).NE.3) THEN

		        nseg = ABS(colr_ndx-last_colr_ndx) 
		        frac = 1./FLOAT(nseg)
		        dx = (xt-xlast)
		        dx = (xt-xlast)* frac
		        dy = (yt-ylast)
		        dy = (yt-ylast)* frac
		        indx = 1
		        inc = 1
		        IF (last_colr_ndx .GT. colr_ndx) inc = -1
		        DO 80 iseg = last_colr_ndx+inc, colr_ndx, inc
			   xi = xlast + dx*indx
			   yi = ylast + dy*indx
	                   line_ndx = iseg + thick_offset + low_index - 1
		           IF (line_ndx .LT. 1) line_ndx = 1
	                   CALL PLOT(xi,yi,IDN,0)
	                   CALL COLOR (line_ndx) 
	                   CALL PLOT(xi,yi,IDN,0)
			   indx = indx + 1
80	                CONTINUE
			XT = xi  ! should already be??
			YT = yi
                      ELSEIF (.NOT.new_branch) THEN
	                IF (ITYPE(I).NE.3) CALL PLOT(xhalf,yhalf,IDN,0)
			IF (colr_ndx.NE.last_colr_ndx) CALL COLOR (line_ndx)
	                IF (ITYPE(I).NE.3) CALL PLOT(xhalf,yhalf,IDN,0)
	                IF (.NOT.new_branch) CALL PLOT(XT,YT,IDN,0)
		     ENDIF
		     IF (colr_ndx.NE.last_colr_ndx) CALL COLOR (line_ndx)
		  ENDIF

	          CALL PLOT(XT,YT,IDN,0)

	          IF (last_colr_ndx .EQ. bad_ribbon_ndx .AND. 
     .		       colr_ndx .NE. bad_ribbon_ndx) THEN
                     CALL COLOR (line_ndx) 
		  ENDIF

		  last_colr_ndx = colr_ndx
		  last_bad = .FALSE.
		  IF (isbad) last_bad = .TRUE.
		  isbad = .FALSE.
	  	  xlast = xt
		  ylast = yt
		  datxlast = datx
		  datylast = daty

		  if (interrupted) goto 51 ! *jd* 8.3.92
		  idec = idec + 1
		  md = MOD(idec,nskp) .EQ. 0
		  IF (bad_blank .AND. colr_ndx.EQ.bad_ribbon_ndx) THEN
		     nothing = 0
		  ELSE
	             IF (md) CALL MARK(IM)
	          ENDIF
		  CALL PLOT(XT,YT,IDN,0) 

	          IF(IDN.EQ.0)THEN
		    xtlast = xt
		    ytlast = yt
		    klast = k
	            IDN=1
	          ENDIF
		  new_branch = .FALSE.
	        END IF

* if gap_var = 1, then this is the end of a segment. Draw to it, and pick up the pen.
		IF (igap.GT.0) THEN
		   IDN = 0
		   CALL PLOT(XT,YT,IDN,0) 
		ENDIF

72	      CONTINUE
	    END IF
	    IF(ITYPE(I).EQ.2.OR.ITYPE(I).EQ.5)CALL MARK(IMARK(I))

	if (interrupted) go to 51 !*jd* 8.3.92


*** FINALIZE
 51	CONTINUE
        CALL PPLGFLUSH
        CALL MAKEDOTFLUSH

        CALL COLOR(1)

*       DO THE KEY
C     Generate a color/gray scale key.
C     Restore standard key style afterward (do this here, to 
C     catch both plot and plot/set calls).
C     
        if (do_key .AND. .NOT.interrupted) THEN
            call key ! 1/89 - *sh*
           CALL SHADE_KEY_CONT (0)
        ENDIF
	RETURN

	END



