	SUBROUTINE FILLPOL(X,NXS,XF,YF,fp_xfac,fp_yfac)
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
C  ARRAY X HOLDS X AND Y'S FOR ALL LINES.  DIMENSIONED FOR NXS+npoly
C  ORDER AS FOLLOWS:  ALL X'S FOR LINE1, ALL X'S FOR LINE2,...
C                     STARTING AT X(NXS/2+1), ALL Y'S FOR LINE1,
C                     ALL Y'S FOR LINE2...
C  XF AND YF ARE FACTORS TO CONVERT TO INCHES FROM USER UNITS
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
*
*
*	*sh* 11/97 based on SHADESUB and PLOTXY
*
* *kob* 3/28/99 - declare subroutine range as external, otherwise F90 tries to
*                 use it as an intrinsic function - more cleanup work
* V510 *sh* 3/00 - changes to accomodate NPOLYGON > NLINES
*		 - involved eliminating CALL TRANS, which means that there
*			isn't a need to copy the X,y coords to x/ypolybase
* V510 4/00 *sh - corrected faulty logic for centered levels (delta wrong)
* V552 3/03 *acm* allow clipping to be set on or off via IWIND (ppl window off)
* V552 3/03 *acm* Check for too many levels, as is done for FILL and CONTOUR plots.
* V553 *acm*  6/03  For polygons, store the # of points in each polygon in the PPLUS
*                   memory buffer, rather than in lleng.  This way the number of
*                   polygons that may be plotted is unlimited. 
* v554 *acm*  2/04  After levels are set, call setlab_syms to set the LEVELS symbols

* v570  4/04*acm* Fix for the section marked correction for rounding errors. 
*                 Use TM_FPEQ_SNGL to see if values are equal or close.  Also fixes 
*                 this adjustment when the range has negative values; was incorrect.
* v580  8/04*acm* Fix bug 957 poly/lev=1 of a constant field needs zlev(1) to be set.
* v581 *acm* 3/05 open levels (-INF) (INF)
* V600 *acm* 8/06 Fix bugs 843, 844: poly/line and reversed axes. Apply xinv and
*                 yinv to coordinates when drawing the lines.
* V62  *acm* 5/09 Fixing bug 1661: if the plot is TRANSPOSED, then apply 
*                 xinv, yinv to the opposite axis.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V72  *acm* 6/17  PyFerret ticket 2542, check for zero-extent on polygons before
*                  calling graphics routines
* V74  *acm* 1/18  Fixing issue 1009: scaling of polygons overlaid on time plot
*                  when rescaling time for precision.
* V745 *acm* 3/19  The fix for 2542 (issue 1814) needs to be applied only for
*                  filled polygons in PyFerret.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'

        INTEGER NXS
	REAL X(*), XF, YF, fp_xfac, fp_yfac

        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'pen_inc.decl'
        include 'PEN.INC'
        include 'pltl_inc.decl'
        include 'PLTL.INC'
        include 'miss_inc.decl'
        include 'MISS.INC'
        include 'hd_inc.decl'
        include 'HD.INC'
        include 'taxis_inc.decl'
        include 'TAXIS.INC'
        include 'axisl_inc.decl'
        include 'AXISL.INC'
        include 'cmrd_inc.decl'
        include 'CMRD.INC'
        include 'cmrdl_inc.decl'
        include 'CMRDL.INC'
        include 'gkscm1_inc.decl'	! sh 11/97
        include 'GKSCM1.INC'		! sh 11/97
!        include 'gkscm2.cmn' ! jd 3.22.89
        include 'gkspar.inc'	! sh 11/97
        include 'shade_vars.cmn'	! sh 11/97
	include		'cont_inc.decl'
	include		'CONT.INC'
        include         'errmsg.parm'
	include 'PPLDAT.INC'		! with ibase
	include 'vector_inc.decl' ! with TRANSPZ
        include 'VECTOR.INC'


* 3/99 *kob*
	EXTERNAL RANGE

	INTEGER xpolybase, ypolybase, zpolybase, polysizemax, nl,
     .		nvertex, fill_ndx, xvertexbase			! 11/97

	INTEGER GET_NDX

* GKS initialization vars
* v491 *kob* removed unnecessary variable declarations
	integer		error,trans_no,cflag
        integer         my_trans
        data            my_trans/20/
	real	 	w(4),v(4),ndc(4),uc(4)
	real		xinv,yinv, tmp

*jd*
	real            zzmin, zzmax, zt
	integer         status, nlm1
	real		dz,znp,zmp, x1val, y1val, xdel, ydel, vlo, vhi
	integer		i,j,uprdn, llen, ptsbase
	logical         line_it, fill_it
        integer         slen1, slen2, ICNT, ii, IY
        CHARACTER       TM_FMT*48, buff1*20, buff2*20, err_msg*40
        real             epsilon
        parameter       (epsilon = 1e-6)

*************************************

C       Check switches
	line_it = sline
	
	if (sline .and. sfill) then
	   fill_it = .true.
	else if (sline .and. .not. sfill) then
	   fill_it = .false.
	else
	   fill_it = .true.
	end if

* INITIALIZE
	nl = lnum - 1	! # lines over which to loop: last line is color info

* determine temp memory buffer location for points translated to plot coords
* ... layout is xline1, xline2, ..., xlinenl, xcolor , bufferspace ...
*               yline1, yline2, ..., ylinenl, ycolor , bufferspace ...
*     "zpolybase" is the offset to the ycolor array
	xpolybase = 1
        ptsbase = ibase - lnum 
	DO 2010 I=1,nl
          llen = x(ptsbase+i-1)
 	  xpolybase = xpolybase+llen
 2010	CONTINUE	

        llen = x(ibase-1)      ! Length of color array

	zpolybase = ptsbase - llen - 1 + nxs/2
	xpolybase = xpolybase + llen	! skip past color array
	ypolybase = xpolybase + nxs/2

        xvertexbase = xpolybase + lnum	! skip past list of lengths

	polysizemax = nxs - ypolybase	! unused buffer space
	fill_ndx = 0

* set up GKS environment (code copied from SHADE_SUB)

*	GET PROPER SCALING, NDC COORDINATES, ETC FOR FILL
	call FGD_GQCNTN(error,trans_no)
	if (error .ne. 0) goto 1100

	call FGD_GQNT(trans_no,error,w,v)
	if (error .ne. 0) goto 1100

	call get_user_coords (w,v,ndc,uc,xinv,yinv)
	
* Fixing bug 1661: if the plot is TRANSPOSED, then apply this
* fix to the opposite axis.

	IF (trnspz) THEN
	   tmp = xinv
	   xinv = yinv
	   yinv = tmp
	ENDIF

*	NOW SET THE CLIPPING RECTANGLE AROUND THEM AND PATTERN SIZE
* v4.91 *kob* had to add <cr> after first line below

	call FGD_GSWN(my_trans,uc(1)*xinv,uc(2)*xinv,uc(3)*yinv,uc(4)*yinv)
        call FGD_GSVP(my_trans,ndc(1),ndc(2),ndc(3),ndc(4))
	if (fill_it) call FGD_GSELNT(my_trans)

	call FGD_GQCLIP(error,cflag,ndc)
	if (error .ne. 0) goto 1000
	if (cflag .NE. iwind) call FGD_GSCLIP(iwind)

*************************************

* original PLOT+ code 
	IBASE=0
	IY=NXS/2
	ZT=YF/XF
	ICNT=1
	IF(M.GE.2.AND.IFLG(2).NE.0)ICNT=P(2)

C *jd* 
* FIND LIMITS
	zmin = 1e33
	zmax = -1e33
	zzmin = 1e33
	zzmax = -1e33
	nlm1  = nl - 1

	do 100 i = 0, nlm1 	      
	   if (ZEQ.AND.x(zpolybase+1+i).EQ.CMZEQ) GOTO 101
	   if (x(zpolybase+1+i) .gt. zzmax) zzmax = x(zpolybase+1+i)
	   if (x(zpolybase+1+i) .lt. zzmin) zzmin = x(zpolybase+1+i)
 101       continue
 100	end do

        IF (zzmin .NE. 1e33) zmin = zzmin  ! Needed for annotating open-ended levels
        IF (zzmax .NE. -1e33) zmax = zzmax

C  Set the contour/color levels
C  ... here we need to call SET_LEVELS_SHADE_FILL if IAUTOC EQ 0 -- levs have been
C      set already.

        IF (IAUTOC .EQ. 0 .OR. 
     .      ((zzmin .NE. 1e33) .AND. (zzmax .NE. -1e33)) ) THEN
           CALL SET_LEVELS_SHADE_FILL (status)
           IF (status .NE. ferr_ok) GOTO 5000
	ELSE
	   IF (sover) THEN
	      IF (zzmin .EQ. 1e33) zmin = 1e36
	      IF (zzmax .EQ. -1e33) zma x= CMZLE - 10
	      CALL SET_LEVELS_SHADE_FILL (status)  !?
	   ENDIF 
	ENDIF

*	DETERMINE PATTERNS OR COLORS TO SET BUNDLED FILL AREA REPRESENTATIONS

*       Allow shading single level fields 
        if (nlev2 .lt. 1) goto 1000
        shd_levels = max (nlev2 - 1,1)

	call set_fill_area_rep (wsid,wstype)

	CALL MARKH(HMARK(1))
	CALL COLOR(IPEN(1))

******** LOOP OVER LINES STORED

	DO 200 I=1,nl
          llen = x(ptsbase+i-1)
* will this polygon fit in the buffer?
	  IF (llen .GT. polysizemax) THEN
	     WRITE (6,*) 'Too much polygon data to plot at once'
	     GOTO 200
	  ENDIF 

* assemble the vertices into arrays in units of plot coordinates
* note: all points in the polygon are guaranteed to be "valid"
	   nvertex = 0
 
* Check that the polygon has a nonzero size in both directions
* PyFerret fails e.g. on a zero-height rectangle. This is not a 
* perfect test but will catch most cases.
* The check is needed only for FILLED polygons.  If using POLY to 
* draw an outline then a zero height or width is ok.

           x1val = x(IBASE+1)
           y1val = x(iy+IBASE+1)
          xdel = 0.
          ydel = 0.

	   DO 150 J=IBASE+1,IBASE+llen
	      nvertex = nvertex + 1
	      x(xvertexbase+nvertex-1) = x(j)
	      x(ypolybase+nvertex-1) = x(iy+j)
              IF (j .GT. ibase+1) THEN
                xdel = MAX( xdel, ABS(x1val-x(j)) )
                ydel = MAX( ydel, ABS(y1val-x(iy+j)) )
              ENDIF

 150	   CONTINUE

	   if (fill_it) then

	      IF (xdel.LT.epsilon .OR. ydel.LT.epsilon) GOTO 157

*             Determine the color index for the fill
	      if (ZEQ.AND.x(zpolybase+i).EQ.CMZEQ) GOTO 157

	      fill_ndx = GET_NDX (x(zpolybase+i),fill_ndx)
	      if (fill_ndx .eq. -999) goto 157

*             Draw the filled polygon
	      CALL SET_FILL_NDX (fill_ndx)

* Rescale the polygons if time-data has been scaled for precision.
* the scaling accounts for the range of the polygon data, needs
* to be widened if overlaying on a longer time axis.

	      IF ( XEQ ) THEN 
	         do 155 j = xvertexbase, xvertexbase+nvertex-1
	            IF (X(J).EQ.CMXEQ) GOTO 157
 155	         continue
	         IF (fp_xfac .NE. 1.) THEN 
	            CALL MINMAX_SNGL( X(xvertexbase), nvertex, CMXEQ, vlo, vhi)
	            do j = xvertexbase, xvertexbase+nvertex-1
	               x(j) = x(j) + (x(j)-vlo)*fp_xfac
	            enddo
	         ENDIF
	      ENDIF

	      IF ( YEQ ) THEN 
	         do 156 j = ypolybase, ypolybase+nvertex-1
	            IF (X(J).EQ.CMYEQ) GOTO 157
 156	         continue
	         IF (fp_yfac .NE. 1.) THEN 
	            CALL MINMAX_SNGL( X(ypolybase), nvertex, CMYEQ, vlo, vhi)
	            do j = xvertexbase, xvertexbase+nvertex-1
	               x(j) = x(j) + (x(j)-vlo)*fp_yfac
	            enddo
	         ENDIF
	      ENDIF

	      if (nvertex .gt. 2) then
		 if (line_it) call FGD_GSELNT(my_trans)
		 CALL FGD_GFA(nvertex, x(xvertexbase), x(ypolybase)) 
	      end if
	   end if

 157	   if (line_it) then
*             Outline the polygon

	      nvertex = 0
	      DO 158 J=IBASE+1,IBASE+llen
		 nvertex = nvertex + 1
		 IF (XEQ.AND.X(xvertexbase+nvertex-1).EQ.CMXEQ) GOTO 199
		 IF (YEQ.AND.X(ypolybase+nvertex-1).EQ.CMYEQ) GOTO 199
 158	      CONTINUE

	      if (fill_it) call FGD_GSELNT(trans_no)
	      nvertex = 0

	      DO 160 J=IBASE+1,IBASE+llen
		 nvertex = nvertex + 1
		 if (nvertex .eq. 1) then
		    uprdn = 0
		 else
		    uprdn = 1
		 end if
		 call plot (xinv*x(xvertexbase+nvertex-1),
     .	                    yinv*x(ypolybase+nvertex-1),uprdn,0)
 160	      continue
	      call plot (xinv*x(xvertexbase), yinv*x(ypolybase),1,0)
	   end if

	   if (interrupted) go to 1000 !*jd* 8.3.92
 199    IBASE=IBASE+llen	! prepare for next line
 200	CONTINUE

******** END OF LOOP OVER LINES STORED


* ... restore GKS transforms following FILLPOL 
 1000   call FGD_GSELNT(trans_no)
        call FGD_GSCLIP(cflag)

* ... and DO THE KEY
	if (fill_it .and. do_key .AND. .NOT.interrupted) then
	   call color(ipen(0))
	   call key
	end if

 1100	CALL COLOR(IPEN(0))

* We use lnum as a counter, but we do not use the lleng array
* or other contstructs that used to treat each polygon as a plot line.
 
        lnum = 1

	RETURN

C  Error exit when too many levels specified.
C  Arrays LWGT, LDIG, and ZLEV are length ICNLEV=500

 5000   ii = ICNLEV
        buff1 = TM_FMT(DBLE(ICNLEV), 0, 20, slen1)
        buff2 = TM_FMT(DBLE(NLEV2), 0, 20, slen2)
        err_msg = '# of levels > '// buff1(1:slen1) //
     .          '  (' // buff2(1:slen2) // ')'
        CALL errmsg (ferr_out_of_range, status, err_msg, *5002)
 5002   return

	END



