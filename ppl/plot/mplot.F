      SUBROUTINE MPLOT(X,NSIZE) 
	DIMENSION X(NSIZE)
C** 
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
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/ZGRID.INC'
	INCLUDE 'pplinc/PLT.INC'
	INCLUDE 'pplinc/AXIS.INC'
	INCLUDE 'pplinc/LABELS.INC'
	INCLUDE 'pplinc/HD.INC'
	INCLUDE 'pplinc/MPLOT.INC'
	INCLUDE 'pplinc/MPLOTX.INC'
	INCLUDE 'pplinc/MPLOTS.INC'
	INCLUDE 'pplinc/TAXIS.INC'
	INCLUDE 'pplinc/XYLABP.INC'
	INCLUDE 'pplinc/FILNAM.INC'
	INCLUDE 'pplinc/LUNITS.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:ZGRID.INC'
        INCLUDE 'pplinc:PLT.INC'
        INCLUDE 'pplinc:AXIS.INC'
        INCLUDE 'pplinc:LABELS.INC'
        INCLUDE 'pplinc:HD.INC'
        INCLUDE 'pplinc:MPLOT.INC'
        INCLUDE 'pplinc:MPLOTX.INC'
        INCLUDE 'pplinc:MPLOTS.INC'
        INCLUDE 'pplinc:TAXIS.INC'
        INCLUDE 'pplinc:XYLABP.INC'
        INCLUDE 'pplinc:FILNAM.INC'
        INCLUDE 'pplinc:LUNITS.INC'
#endif
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
10	XORG=XORG+XLENM(I-1)+DXM(I) 
	YORG=DYM(IYM) 
	DO 20 I=IYM+1,MNY 
20	YORG=YORG+DYM(I)+YLENM(I) 
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
#ifndef unix
	OPEN(LMULT,FILE=MULTFL,STATUS='OLD',
     *	ACCESS='APPEND',FORM='UNFORMATTED')
#endif
	WRITE(LMULT)IPTS,MXBOT,MYLEF
	WRITE(LMULT)XLABS,YLABS,RLABS,LTYPE,XLLABS,YLLABS,LLTYPE,
     *	RHLABS,USRLBS
	CALL SAVE(LMULT)
C
	IF(IPTS(11).EQ.1)THEN
	CALL SAVEXY(LMULT,X,NSIZE)
	ELSE
C
	CALL SAVEZ(LMULT,X,IPTS(7),IPTS(10))
	ENDIF
	IF(IZTYP.NE.0)CALL SAVEZG(LMULT)
C
#ifndef unix
	CLOSE(LMULT)
#endif
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
	SUBROUTINE SAVEZ(LN,Z,NXS,NYS)
	DIMENSION Z(NXS,NYS)
#ifdef unix
	include 'pplinc/PARAMPL5.DAT'
	include 'pplinc/HD.INC'
	include 'pplinc/PPLDAT.INC'
#else
	include 'pplinc:parampl5.dat'
	include 'pplinc:hd.inc'
	include 'pplinc:ppldat.inc'
#endif
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
#ifdef unix
	INCLUDE 'pplinc/PARAMPL5.DAT'
	INCLUDE 'pplinc/LINES.INC'
#else
        INCLUDE 'pplinc:PARAMPL5.DAT'
        INCLUDE 'pplinc:LINES.INC'
#endif
	DIMENSION X(NSIZE)
	IBASE=0
	IYB=NSIZE/2
	DO 6002 I=1,LNUM
	WRITE(LN)(X(IBASE+J),X(IYB+J),J=1,LLENG(I))
	IBASE=IBASE+LLENG(I)
6002	IYB=IYB+LLENG(I)
	RETURN
	ENTRY GETXY(LN,X,NSIZE)
	IBASE=0
	IYB=NSIZE/2
	DO 6001 I=1,LNUM
	READ(LN,END=200)(X(IBASE+J),X(IYB+J),J=1,LLENG(I))
	IBASE=IBASE+LLENG(I)
6001	IYB=IYB+LLENG(I)
200	RETURN
	END
