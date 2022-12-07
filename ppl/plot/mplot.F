      SUBROUTINE MPLOT(X,NSIZE) 
	DIMENSION X(NSIZE)
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


* V54: *acm* 9/01 changes for dynamic allocation of PPLUS memory.
*                 Pass array X to SAVEZ
*      *acm* 10/01  remove VMS includes
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.
*
C**    @(#)mplot.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'ZGRID.INC'
	INCLUDE 'PLT.INC'
	INCLUDE 'AXIS.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'HD.INC'
	INCLUDE 'MPLOT.INC'
	INCLUDE 'MPLOTX.INC'
	INCLUDE 'MPLOTS.INC'
	INCLUDE 'TAXIS.INC'
	INCLUDE 'XYLABP.INC'
	INCLUDE 'FILNAM.INC'
	INCLUDE 'LUNITS.INC'

C     SAVE ORGS AND HEIGHTS 
	IF(IMCNT.EQ.0)THEN 
	    XORGT=XORG 
	    YORGT=YORG 
	    XLENT=XLEN 
	    YLENT=YLEN 
	END IF 
	IMCNT=IMCNT+1 
	IXM=IXM+1 
	IF(IXM.GT.MNX)THEN 
	    IXM=1 
	    IYM=IYM+1 
	END IF 
C     CALC XORG AND YORG 
	XORG=DXM(1) 
	DO 10 I=2,IXM 
	   XORG=XORG+XLENM(I-1)+DXM(I) 
10	CONTINUE	
	YORG=DYM(IYM) 
	DO 20 I=IYM+1,MNY 
	   YORG=YORG+DYM(I)+YLENM(I) 
20	CONTINUE	
	XLEN=XLENM(IXM) 
	YLEN=YLENM(IYM) 
	LABYT=LABELY 
	LABXT=LABELX 
	LABT=LAB1 
	XLABT=XLAB 
	YLABT=YLAB 
C     SET AXIS LABELLING 
	IF(DXM(IXM).EQ.0)THEN 
	    MYLEF=1 
	    IF(LABELY.NE.0)THEN 
		IF(DXM(IXM+1).EQ.0.)THEN 
		    LABELY=0 
		    YLAB=' ' 
		ELSE 
		    LABELY=1 
		END IF 
	    END IF 
	END IF 
	IF(DYM(IYM).EQ.0)THEN 
	    LAB1=' ' 
	    MXBOT=1 
	    IF(LABELX.NE.0)THEN 
		LAB1=' ' 
		IF(IYM.EQ.1)THEN 
		    LABELX=1 
		ELSE IF(DYM(IYM-1).EQ.0.)THEN 
		    LABELX=0 
		    XLAB=' ' 
		ELSE 
		LABELX=1 
		END IF 
	    END IF 
	END IF 
C     SAVE PLOT ON FILE 
	WRITE(LMULT)IPTS,MXBOT,MYLEF
	WRITE(LMULT)XLABS,YLABS,RLABS,LTYPE,XLLABS,YLLABS,LLTYPE,
     *	RHLABS,USRLBS
	CALL SAVE(LMULT)
C
	IF(IPTS(11).EQ.1)THEN
	CALL SAVEXY(LMULT,X,NSIZE)
	ELSE
C
	CALL SAVEZ(LMULT,X,IPTS(7),IPTS(10),X)
	ENDIF
	IF(IZTYP.NE.0)CALL SAVEZG(LMULT)
C
	IVFLG=0
	LAB1=LABT
      XLAB=XLABT 
      YLAB=YLABT 
      LABELX=LABXT 
      LABELY=LABYT 
      MXBOT=0 
      MYLEF=0 
      RETURN 
      ENTRY MPLOT2(X,NSIZE)
      IMULT=0 
      IMCNT=0 
      MNX=0 
      MNY=1 
      XORG=XORGT 
      YORG=YORGT 
      HEIGHT=HTM 
      WIDTH=WDM 
      XLEN=XLENT 
      YLEN=YLENT 
      MXBOT=0 
      MYLEF=0 
      LAB1=LABT 
      XLAB=XLABT 
      YLAB=YLABT 
      LABELX=LABXT 
      LABELY=LABYT 
      RETURN 
      END 

	SUBROUTINE SAVEZ(LN,Z,NXS,NYS,X)
	DIMENSION Z(NXS,NYS),X(*)

	include 'PARAMPL5.DAT'
	include 'HD.INC'
	include 'PPLDAT.INC'

        IF(JTYPE.EQ.0)THEN
            WRITE(LN)Z
        ELSE IF(JTYPE.EQ.-1)THEN
            WRITE(LN)(X(K),K=1,2*NXS*NYS)
        ELSE IF(JTYPE.EQ.2)THEN
            WRITE(LN)(X(K),K=1,NXS*NYS+NXS+NYS)
        ELSE
            NXNY2=2*(NXS*NYS+NXS+NYS)
            WRITE(LN)(X(K),K=1,NXNY2)
        ENDIF
        RETURN
        ENTRY GETZ(LN,Z,NXS,NYS)
        IF(JTYPE.EQ.0)THEN
            READ(LN,END=200)Z
        ELSE IF(JTYPE.EQ.-1)THEN
            READ(LN,END=200)(X(K),K=1,2*NXS*NYS)
        ELSE IF(JTYPE.EQ.2)THEN
            READ(LN,END=200)(X(K),K=1,NXS*NYS+NXS+NYS)
        ELSE
            NXNY2=2*(NXS*NYS+NXS+NYS)
            READ(LN,END=200)(X(K),K=1,NXNY2)
        ENDIF
200     RETURN
	END

	SUBROUTINE SAVEXY(LN,X,NSIZE)
	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'LINES.INC'
	DIMENSION X(NSIZE)
	IBASE=0
	IYB=NSIZE/2
	DO 6002 I=1,LNUM
	  WRITE(LN)(X(IBASE+J),X(IYB+J),J=1,LLENG(I))
	  IBASE=IBASE+LLENG(I)
	  IYB=IYB+LLENG(I)
6002	CONTINUE
	RETURN

	ENTRY GETXY(LN,X,NSIZE)
	IBASE=0
	IYB=NSIZE/2
	DO 6001 I=1,LNUM
	   READ(LN,END=200)(X(IBASE+J),X(IYB+J),J=1,LLENG(I))
	   IBASE=IBASE+LLENG(I)
	   IYB=IYB+LLENG(I)
6001	CONTINUE

200	RETURN
	END
