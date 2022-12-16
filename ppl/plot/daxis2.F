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
*     MOD *JD* 10.91 to eliminate PLOT COMPLETE msg if in ferret
*     MOD *sh* 6/99 with changes to IGRNPT call
* v552 *acm* 3/03 Changes for multi-line labels.
* V69+  5/14 *acm* Fix for ascii font in multi-line labels, ticket 2176

* V695  4/15 (PyFerret) Logo labels drawn with the logo text style
* 12/2022 *acm* Code cleanup: remove code having to do with Tektronix
*		 and other terminals. 
* 12/2022 *acm* Code cleanup: remove obsolete multplt

	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'AXIS.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'PLT.INC'
	INCLUDE 'LINES.INC'
	INCLUDE 'TAXIS.INC'
	INCLUDE 'TICS.INC'
	INCLUDE 'TICS2.INC'
	INCLUDE 'AXISL.INC'
	INCLUDE 'XYLABP.INC'
	INCLUDE 'CMRDL.INC'
	INCLUDE 'LUNITS.INC'
	INCLUDE 'PRMCOM.INC'
	INCLUDE 'SYSTEM.INC'
	INCLUDE 'SWITCH.INC'

        include 'xinterrupt.cmn' ! 1/89 *sh*
        include 'ppl_in_ferret.cmn'
        include 'pyfonts.cmn'

******************************************************************************


	CHARACTER OUT*255
        INTEGER TM_LENSTR, numlines, i, j
        CHARACTER labeltmpstr*2048
        INTEGER labeltmpstrlen

c        CHARACTER fontprefix*3, penprefix*3, newfont*3, newpen*3
c        INTEGER lnbeg(500), lnend(500), 
c        REAL line_spacing
c        line_spacing = 1.4

	CALL VECTRS
	DO 80 I=1,NLAB

* Don't go through the motions of plotting blank labels
	IF (TM_LENSTR(labs(i)) .EQ. 0) GOTO 80

	IF (pyfont .AND. pyf_set(ilogo)) THEN
	   IF (logolab(1).EQ.i .OR. logolab(2).EQ.i .OR. 
     .      logolab(3).EQ.i) igroup = ilogo
        ENDIF

	IF(RHLABS(I).LT.0.0)THEN
		HLB=HLABS
	ELSE
		HLB=RHLABS(I)
	ENDIF
	CS=COS(RLABS(I)*0.0175329252)
	SN=SIN(RLABS(I)*0.0175329252)
	IAT=0
	SHFT=0.0


        CALL tm_break_lines (labs(i), lnbeg, lnend, numlines)
        nfont = 0
        npen = 0
        fontprefix = ""
        penprefix = ""
        DO 20 j = 1, numlines
           CALL getlabfonts (labs(i), lnbeg, lnend, j, fontprefix, 
     .           npnew, nfnew, newfont, newpen, ncr)

	   NCR = lnend(j) - lnbeg(j) + 1
C          Save to a temporary string to fix valgrid-reported issue with
C          using uninitialized characters
           IF ( (npen .GT. 0) .AND. (nfont .GT. 0) ) THEN
              labeltmpstr = 
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = npen + nfont + NCR
           ELSE IF ( npen .GT. 0 ) THEN
              labeltmpstr = 
     .                 penprefix(1:npen)//
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = npen + NCR
           ELSE IF ( nfont .GT. 0 ) THEN
              labeltmpstr = 
     .                 fontprefix(1:nfont)//
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = nfont + NCR
           ELSE
              labeltmpstr = 
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = NCR
           ENDIF
	   XMN=SYMWID(HLB, labeltmpstrlen, labeltmpstr)

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
     *	       XLLABS(I)/XFUSER+XLOUSR,YLLABS(I)/YFUSER+YLOUSR,
     *	       LLTYPE(I),SHFT,XMN,CS,SN,XF,YF,HLB)
	       GOTO 70
	   ENDIF
	   ICHAR=IGRNPT(.TRUE., XL,YL)	! show the choice menu
	   LTY=-1
	   IF(ICHAR.EQ.67.OR.ICHAR.EQ.99)THEN
	       LTY=0
	       SHFT=-XMN*.5
	   ELSE IF(ICHAR.EQ.82.OR.ICHAR.EQ.114)THEN
	       LTY=1
	       SHFT=-XMN
	   ELSE IF(ICHAR.EQ.76.OR.ICHAR.EQ.70.OR.ICHAR.EQ.108
     *	   .OR.ICHAR.EQ.102)THEN
	       IAT=1
	       XL2=XL
	       YL2=YL
	       ICHARO=ICHAR
	       ICHAR=IGRNPT(.FALSE., XL,YL) ! do not show the choice menu again
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
	       WRITE(OUT,998)I,(XL2-XLOUSR)*XFUSER,
     .               (YL2-YLOUSR)*YFUSER,IAT
998 	       FORMAT('C LLABS,',I2,2(',',1PE10.3),',',I2)
	       CALL ECHO(OUT,40)
	   ENDIF
	   XL=(XL-XLOUSR)*XFUSER
  	   YL=(YL-YLOUSR)*YFUSER
	   IF(USRLBS(I))THEN
	       WRITE(OUT,999)I,XL,YL,LTY,LABS(I)(:NCR)
999	       FORMAT('C LABS,',I2,2(',',1PE10.3),',',I2,',','"',A,'"')
	   ELSE
	       WRITE(OUT,997)I,XL,YL,LTY,LABS(I)(:NCR)
997	       FORMAT('C LABS/NOUSER,',I2,2(',',1PE10.3),',',I2,
     *                ',','"',A,'"')
	   ENDIF
	   NOUT=LNBLK(OUT,255)
	   CALL ECHO(OUT,NOUT)
70	   XL=XL/XFUSER+XLOUSR+SHFT*CS/XF + 
     .         line_spacing*(j-1)*hlb*SN/xf
	   YL=YL/YFUSER+YLOUSR+SHFT*SN/YF -
     .         line_spacing*(j-1)*hlb*CS/yf

C          Save to a temporary string to fix valgrid-reported issue with
C          using uninitialized characters
           IF ( (npen .GT. 0) .AND. (nfont .GT. 0) ) THEN
              labeltmpstr = 
     .                 penprefix(1:npen)//fontprefix(1:nfont)//
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = npen + nfont + NCR
           ELSE IF ( npen .GT. 0 ) THEN
              labeltmpstr = 
     .                 penprefix(1:npen)//
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = npen + NCR
           ELSE IF ( nfont .GT. 0 ) THEN
              labeltmpstr = 
     .                 fontprefix(1:nfont)//
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = nfont + NCR
           ELSE
              labeltmpstr = 
     .                 LABS(I)(lnbeg(j):lnend(j))
              labeltmpstrlen = NCR
           ENDIF
           CALL SYMBEL(XL,YL,RLABS(i),HLB,labeltmpstrlen,labeltmpstr)

C  Set up to use any font settings from this line on subsequent lines.

           IF (npnew.GT.0) THEN
              penprefix(1:3) = newpen(1:3)
              npen = 3
           ENDIF
           IF (nfnew.GT.0) THEN
              fontprefix(1:3) = newfont(1:3)
              nfont = 3
           ENDIF

20      CONTINUE

	IF (pyfont) igroup = fnt_all
80	CONTINUE

	IF(PLTOPN)RETURN
	CALL HABIS
	call ferret_plot_complete (icount)

	RETURN
	END

