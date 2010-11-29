        PROGRAM DRWSYM
        CHARACTER FONT*2,NUM*6,STR*5,NAME*30,NME*30
	DIMENSION ILEN
	DATA ILEN/88/
	DATA FONT/'MK'/
	DATA NAME/'Plot Marks'/
        CALL PLTYPE(0)
        CALL SIZE(8.2,10.5)
        SZE=0.21
        CALL ERASE
C
C       DRAW BOX
C
        X=0.
        Y1=7.5
        Y2=0.
        DO 110 I=1,10
        CALL PLOT(X,Y1,0,0)
        CALL PLOT(X,Y2,1,0)
        X=X+0.75
        IF(Y1.EQ.0.)THEN
                Y2=Y1
                Y1=7.5
        ELSE
                Y2=Y1
                Y1=0.
        ENDIF
110     CONTINUE
        Y=7.5
        X1=0.
        X2=6.75
        DO 120 I=1,11
        CALL PLOT(X1,Y,0,0)
        CALL PLOT(X2,Y,1,0)
        Y=Y-0.75
        IF(X1.EQ.0.)THEN
                X2=X1
                X1=6.75
        ELSE
                X2=X1
                X1=0.
        ENDIF
120     CONTINUE
C
C       DRAW CHARACTERS
C
        WRITE(STR,999)FONT
999     FORMAT('@',A2)
        WRITE(NME,997)NAME
997     FORMAT('@SR',A25)
        CALL SYMBEL(0.,7.6,0.0,.14,28,NME)
	ICH=1
        DO 130 I=1,9
        DO 130 J=1,10
        X=(I-1)*0.75
        Y=7.5-J*0.75
        WRITE(NUM,998)ICH
998     FORMAT('@SR',I2.2)
        STR(4:5)=NUM(4:5)
        CALL SYMBEL(X+0.02,Y+0.65,0.,0.07,6,NUM)
        S=X+.375-SYMWID(SZE,5,STR)*0.5
        CALL SYMBEL(S,Y+0.270,0.,SZE,5,STR)
	ICH=ICH+1
	IF(ICH.GT.ILEN)GOTO 140
130     CONTINUE
140     CALL PLTEND
        CALL EXIT(0)
        END
