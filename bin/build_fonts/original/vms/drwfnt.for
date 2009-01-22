        PROGRAM DRWFNT
        CHARACTER FONT(21)*2,NUM*6,STR*4,NAME(21)*30,NME*30
        CHARACTER NA*7,CCHR*1
        BYTE BCHR
        EQUIVALENCE (BCHR,CCHR)
        DATA FONT/'SR','DR','TR','CR','AS','AC','CS','TI','GE',
     *  'IR','SS','CI','II','SG','CG','IG','GG','GI',
     *  'CC','AR','AG'/
        DATA NAME/'Simplex Roman','Duplex Roman','Triplex Roman',
     *  'Complex Roman','ASCII Simplex Roman','ASCII Complex Roman',
     *	'Complex Script','Triplex Italic',
     *  'Gothic English','Indexical Complex Roman',
     *  'Simplex Script','Complex Italic',
     *  'Indexical Complex Italic','Simplex Greek','Complex Greek',
     *  'Indexical Complex Greek','Gothic German','Gothic Italian',
     *  'Complex Cyrillic','Cartographic Roman',
     *  'Cartographic Greek'/
        DATA NA/'@SRn.a.'/
        OPEN(5,FILE='TI:',STATUS='UNKNOWN')
        CALL PLTYPE(0)
        CALL SIZE(10.5,8.2)
        SZE=0.21
        DO 100 IFNT=21,1,-1
        CALL ERASE
C
C       DRAW BOX
C
        X=5.5
        Y1=8.25
        Y2=0.25
        DO 110 I=1,7
        CALL PLOT(Y1,X,0,0)
        CALL PLOT(Y2,X,1,0)
        X=X-0.75
        IF(Y1.EQ.0.25)THEN
                Y2=Y1
                Y1=8.25
        ELSE
                Y2=Y1
                Y1=0.25
        ENDIF
110     CONTINUE
        Y=8.25
        X1=5.5
        X2=1.0
        DO 120 I=1,17
        CALL PLOT(Y,X1,0,0)
        CALL PLOT(Y,X2,1,0)
        Y=Y-0.5
        IF(X1.EQ.1.0)THEN
                X2=X1
                X1=5.5
        ELSE
                X2=X1
                X1=1.0
        ENDIF
120     CONTINUE
C
C       DRAW CHARACTERS
C
        WRITE(STR,999)FONT(IFNT)
999     FORMAT('@',A2)
        WRITE(NME,997)NAME(IFNT)
997     FORMAT('@SR',A25)
        CALL SYMBEL(8.35,5.5,-90.0,.14,28,NME)
        DO 130 I=1,6
        DO 130 J=1,16
        ICH=(I-1)*16+J+31
        X=5.5-(I-1)*0.75
        Y=8.25-J*0.5
        IF((IFNT.NE.5.AND.IFNT.NE.6).AND.
     *	    (ICH.EQ.92.OR.ICH.EQ.94.OR.ICH.EQ.95.OR.ICH.EQ.64))THEN
                S=X-0.375+SYMWID(.10,7,NA)*0.5
                CALL SYMBEL(Y+0.2,S,-90.,.10,7,NA)
        ELSE
                STR(4:4)=CHAR(ICH)
                WRITE(NUM,998)ICH
998             FORMAT('@SR',I3)
                CALL SQUISH(NUM,1,6)
                CALL SYMBEL(Y+0.4,X-0.02,-90.,0.07,6,NUM)
                S=X-0.37+SYMWID(SZE,4,STR)*0.5
                CALL SYMBEL(Y+0.15,S,-90.,SZE,4,STR)
        ENDIF
130     CONTINUE
        CALL PLTEND
100     CONTINUE
        CALL EXIT
        END
