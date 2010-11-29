        PROGRAM DRWSYM
        CHARACTER FONT(20:30)*2,NUM*6,STR*5,NAME(20:30)*30,NME*30
	DIMENSION ILEN(20:30)
	DATA ILEN/26,69,15,12,42,43,78,19,39,32,27/
C	DATA ILEN/27,32,39,19,78,43,42,12,15,69,26/
	DATA FONT/'CA','IM','IZ','LM','MP','SM','MA',
     *	'WE','EL','MU','ZO'/
C        DATA FONT/'ZO','MU','EL','WE','MA','SM','MP',
C     *  'LM','IZ','IM','CA'/
	DATA NAME/'Cartographic','Indexical Math','Indexical Zodiac',
     *	'Large Math','Map','Simplex Math','Math','Weather','Electrical',
     *	'Music','Zodiac'/
c        DATA NAME/'Zodiac','Music','Electrical','Weather',
c     *	'Math','Simplex Math','Map','Large Math','Indexical Zodiac',
c     *	'Indexical Math','Cartographic'/
        OPEN(5,FILE='TI:',STATUS='UNKNOWN')
        CALL PLTYPE(0)
        CALL SIZE(10.5,8.2)
        SZE=0.21
        DO 100 IFNT=20,30
        CALL ERASE
C
C       DRAW BOX
C
        X=6.50
        Y1=9.00
        Y2=0.25
        DO 110 I=1,9
        CALL PLOT(Y1,X,0,0)
        CALL PLOT(Y2,X,1,0)
        X=X-0.875
        IF(Y1.EQ.0.25)THEN
                Y2=Y1
                Y1=9.00
        ELSE
                Y2=Y1
                Y1=0.25
        ENDIF
110     CONTINUE
        Y=9.00
        X1=6.50
        X2=-0.50
        DO 120 I=1,11
        CALL PLOT(Y,X1,0,0)
        CALL PLOT(Y,X2,1,0)
        Y=Y-0.875
        IF(X1.EQ.-0.50)THEN
                X2=X1
                X1=6.50
        ELSE
                X2=X1
                X1=-0.50
        ENDIF
120     CONTINUE
C
C       DRAW CHARACTERS
C
        WRITE(STR,999)FONT(IFNT)
999     FORMAT('@',A2)
        WRITE(NME,997)NAME(IFNT)
997     FORMAT('@SR',A25)
        CALL SYMBEL(9.10,6.50,-90.0,.14,28,NME)
	ICH=1
        DO 130 I=1,8
        DO 130 J=1,10
        X=6.50-(I-1)*0.875
        Y=9.00-J*0.875
        WRITE(NUM,998)ICH
998     FORMAT('@SR',I2.2)
        STR(4:5)=NUM(4:5)
        CALL SYMBEL(Y+0.775,X-0.02,-90.,0.07,6,NUM)
        S=X-0.4375+SYMWID(SZE,5,STR)*0.5
        CALL SYMBEL(Y+0.3325,S,-90.,SZE,5,STR)
	ICH=ICH+1
	IF(ICH.GT.ILEN(IFNT))GOTO 140
130     CONTINUE
140     CALL PLTEND
100     CONTINUE
        CALL EXIT
        END
