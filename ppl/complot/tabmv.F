      SUBROUTINE TABMV
C** 
C**    @(#)tabmv.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	INTEGER*4 ITA,ITB
C
C     CALL TABMV
C
C     THIS SUBROUTINE DRIVES THE TEKTRONIX GRAPHICS TERMINAL.  THE
C     COORDINATES OF THE NEW POSITION ARE RECEIVED IN VARIABLES A
C     AND B.
C
C
#ifdef unix
	INTEGER*2 DUMMY(4)
#else
	byte dummy(4)
#endif
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
	DATA DUMMY/32,96,32,64/
C
C     CONVERT THE POSITION TO A TEKTERMINAL ADDRESS.
C
      ITA = IFIX(A * PLTOTA)
      ITB = IFIX(B * PLTOTB)
C
C     CHECK THE GRAPHICS FLAG.
C
        IF(GRAPHF)THEN
C
C     CHECK THE PEN POSITION.
C
          IF(.NOT. PEN)CALL CHOUT(GS,1)
        ELSE
C
C     CHECK THE TEKTERMINAL TYPE.
C
          IF(TTYPE .EQ. -4662 .OR. TTYPE .EQ. -4663)THEN
C
C     TRANSMIT THE 'PLOTTER ON' COMMAND TO THE TEKTRONIX PLOTTER.
C
            CALL CHOUT(TEKON,3)
            OLDHTA = 0
            OLDHTB = 0
          ENDIF
C
C     SELECT VECTORS MODE.
C
          CALL CHOUT(GS,1)
          LHIA = 0
          LHIB = 0
          LLOBA = 0
C
C     SET THE GRAPHICS FLAG.
C
          GRAPHF = .TRUE.
C
C     CHECK THE PEN POSITION.
C
          IF(PEN)THEN
C
C     PLOT A DARK LINE SEGMENT SO THAT THE NEXT LINE SEGMENT
C     PLOTTED WILL BE BRIGHT.
C
            CALL CHOUT(DUMMY,4)
          ENDIF
        ENDIF
C
C     CHECK THE TEKTERMINAL TYPE.
C
      IF(TTYPE .GT. 0 .AND. TTYPE .LT. 4014)THEN
C
C     CALCULATE THE HIGH-ORDER B-BYTE.  10 BIT CODES FOR TEK401X
C
#ifdef unix
#ifdef sun
	ihib=or(and(rshift(itb,5),31),32)
#else
# ifdef NEED_IAND
	IHIB=IOR(IAND(ISHFT(ITB,-5),31),32)
# else
	IHIB=OR(AND(LSHIFT(ITB,-5),31),32)
# endif
#endif
#else
	IHIB=IOR(IAND(ISHFT(ITB,-5),31),32)
#endif
C
C     IF NO CHANGE, DO NOT TRANSMIT IT.
C
        IF(IHIB .NE. LHIB)THEN
          CALL CHOUT(IHIB,1)
          LHIB = IHIB
        ENDIF
C
C     TRANSMIT THE LOW-ORDER B-BYTE.
C
#ifdef unix
# ifdef NEED_IAND
	CALL CHOUT(IOR(IAND(ITB,31),96),1)
# else
	CALL CHOUT(OR(AND(ITB,31),96),1)
# endif
#else
	CALL CHOUT(IOR(IAND(ITB,31),96),1)
#endif
C
C     CALCULATE THE HIGH-ORDER A-BYTE.
C
#ifdef unix
#ifdef sun
	IHIA=OR(AND(rSHIFT(ITA,5),31),32)
#else
# ifdef NEED_IAND
	IHIA=IOR(IAND(ISHFT(ITA,-5),31),32)
# else
	IHIA=OR(AND(LSHIFT(ITA,-5),31),32)
# endif
#endif
#else
	IHIA=IOR(IAND(ISHFT(ITA,-5),31),32)
#endif
C
C     IF NO CHANGE, DO NOT TRANSMIT IT.
C
        IF(IHIA .NE. LHIA)THEN
          CALL CHOUT(IHIA,1)
          LHIA = IHIA
        ENDIF
C
C     TRANSMIT THE LOW-ORDER A-BYTE.
C
#ifdef unix
# ifdef NEED_IAND
	CALL CHOUT(IOR(IAND(ITA,31),64),1)
# else
	CALL CHOUT(OR(AND(ITA,31),64),1)
# endif
#else
	CALL CHOUT(IOR(IAND(ITA,31),64),1)
#endif
      ELSE
        ICH = 3
C
C     CALCULATE THE HIGH-ORDER B-BYTE.    12 BIT CODES FOR TEK41XX
C
#ifdef unix
#ifdef sun
        IHIB = OR(AND(rSHIFT(ITB,7), 31), 32)
#else
# ifdef NEED_IAND
        IHIB = IOR(IAND(ISHFT(ITB,-7), 31), 32)
# else
        IHIB = OR(AND(LSHIFT(ITB,-7), 31), 32)
# endif
#endif
#else
        IHIB = IOR(IAND(ISHFT(ITB,-7), 31), 32)
#endif
C
C     IF NO CHANGE, DO NOT TRANSMIT IT.
C
        IF(IHIB .NE. LHIB)THEN
          CALL CHOUT(IHIB,1)
          LHIB = IHIB
          ICH = ICH + 1
        ENDIF
C
C     CALCULATE THE EXTRA BYTE.
C
#ifdef unix
# ifdef NEED_IAND
        ILOBA=IOR(IOR(ISHFT(IAND(ITB,3),2),IAND(ITA,3)),96)
# else
        ILOBA=OR(OR(LSHIFT(AND(ITB,3),2),AND(ITA,3)),96)
# endif
#else
        ILOBA=IOR(IOR(ISHFT(IAND(ITB,3),2),IAND(ITA,3)),96)
#endif
C
C     IF NO CHANGE, DO NOT TRANSMIT IT.
C
        IF(ILOBA .NE. LLOBA)THEN
          CALL CHOUT(ILOBA,1)
          LLOBA = ILOBA
          ICH = ICH + 1
        ENDIF
C
C     TRANSMIT THE LOW-ORDER B-BYTE.
C
#ifdef unix
#ifdef sun
        CALL CHOUT(OR(AND(rSHIFT(ITB,2), 31), 96),1)
#else
# ifdef NEED_IAND
        CALL CHOUT(IOR(IAND(ISHFT(ITB,-2), 31), 96),1)
# else
        CALL CHOUT(OR(AND(LSHIFT(ITB,-2), 31), 96),1)
# endif
#endif
#else
        CALL CHOUT(IOR(IAND(ISHFT(ITB,-2), 31), 96),1)
#endif
C
C     CALCULATE THE HIGH-ORDER A-BYTE.
C
#ifdef unix
#ifdef sun
        IHIA = OR(AND(rSHIFT(ITA,7), 31), 32)
#else
# ifdef NEED_IAND
        IHIA = IOR(IAND(ISHFT(ITA,-7), 31), 32)
# else
        IHIA = OR(AND(LSHIFT(ITA,-7), 31), 32)
# endif
#endif
#else
        IHIA = IOR(IAND(ISHFT(ITA,-7), 31), 32)
#endif
C
C     IF NO CHANGE, DO NOT TRANSMIT IT.
C
        IF(IHIA .NE. LHIA)THEN
          CALL CHOUT(IHIA,1)
          LHIA = IHIA
          ICH = ICH + 1
        ENDIF
C
C     TRANSMIT THE LOW-ORDER A-BYTE.
C
#ifdef unix
#ifdef sun
        CALL CHOUT(OR(AND(rSHIFT(ITA,2), 31), 64),1)
#else
# ifdef NEED_IAND
        CALL CHOUT(IOR(IAND(ISHFT(ITA,-2), 31), 64),1)
# else
        CALL CHOUT(OR(AND(LSHIFT(ITA,-2), 31), 64),1)
# endif
#endif
#else
        CALL CHOUT(IOR(IAND(ISHFT(ITA,-2), 31), 64),1)
#endif
C
C     CHECK THE TEKTERMINAL TYPE.
C
        IF(TTYPE .EQ. -4662 .OR. TTYPE .EQ. -4663)THEN
C
C     TRANSMIT SYNC CHARACTERS WHILE THE PLOTTER DRAWS THE VECTOR.
C
          DA = TWIDTH * FLOAT(ITA - OLDHTA) / TSCLA
          DB = THIGHT * FLOAT(ITB - OLDHTB) / TSCLB
          OLDHTA = ITA
          OLDHTB = ITB
C
C     NOMINAL VECTOR DRAWING TIMES FOR THE TEKTRONIX PLOTTER CAN
C     BE DETERMINED AS FOLLOWS:
C        T (IN MSEC) = 61.44 * L (IN INCHES) + 84.93
C
          NDELAY = IFIX((0.006144 * SQRT(DA * DA + DB * DB) +
     *     0.0050958) * FLOAT(BAUDR))
          IF(NDELAY .GE. ICH)THEN
            DO 10 J=ICH,NDELAY
   10       CALL CHOUT(SYN,1)
          ENDIF
        ENDIF
      ENDIF
      RETURN
C
      END




