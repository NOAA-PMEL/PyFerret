      program mark
C** 
C**    @(#)mark.F	1.3     5/1/89
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
      CHARACTER IMODE*1
      LOGICAL LARGE
      integer sx(1000),sy(1000),ascii(128),ipnt
C
C     THE NEXT LINE IS MACHINE DEPENDENT AND MAY BE REPLACED BY
      INTEGER*2 MRKTAB(2,44),TABT(200)
C
C
      DATA MRKTAB/    5,   9,  11,  15,  14,  15,  11,  12,
     *               26,  31,  32,  37,  38,  43,  44,  49,
     *                1,   5,  64,  67,   5,  15,  50,  54,
     *                1,   9,  55,  63,  15,  19,  21,  25,
     *               50,  53,  51,  54,  72,  77,  84,  98,
     *               18,  22,  11,  19,  64,  66,  68,  71,
     *               68,  70,  78,  83, 102, 106, 113, 118,
     *              119, 124, 125, 130, 131, 136, 105, 110,
     *              107, 112, 137, 139,  99, 106, 103, 108,
     *              140, 144, 140, 147, 156, 163, 148, 155,
     *              170, 183, 184, 189, 188, 193, 164, 169/
C
	DATA TABT/9,41,45,13,9,45,0,13,41,0,
     *25,29,0,11,43,29,11,25,43,11,25,29,11,43,29,18,27,34,0,27,
     *24,20,27,36,0,27,30,20,27,18,0,3,27,36,27,34,0,27,51,41,
     *13,45,9,41,4,2,16,32,50,52,38,22,4,9,29,41,9,13,25,45,
     *13,13,27,31,0,27,45,9,27,29,0,27,41,13,20,18,9,0,18,34,
     *0,20,36,0,45,36,34,41,19,35,0,21,17,33,37,21,19,35,33,17,
     *21,37,20,29,25,0,17,33,21,37,35,19,17,33,21,37,19,35,33,17,
     *21,19,43,0,37,33,21,37,25,12,44,0,42,10,0,17,37,26,30,0,
     *12,44,0,8,40,13,45,0,43,11,0,9,41,4,41,30,9,52,4,12,
     *20,21,13,12,0,9,45,0,33,41,42,34,33,14,44,10,0,9,41,0,
     *42,12,46,0,0,0,0,0,0,0/
C
C     CHECK THE DATA MARK CODE.
C
	ipnt=1
	ibase=-3
	irast=7
	do 100 imark=1,88
C
C     DETERMINE THE SIZE OF THE DATA MARK AND ITS CODE.
C
      JMARK = (IMARK + 1) / 2
      LARGE = .FALSE.
      IF(2 * JMARK .EQ. IMARK)LARGE = .TRUE.
C
C     GET CONTROL INFORMATION FROM THE POINTER TABLE.
C
      IPOINT = MRKTAB(1,JMARK)
      ILAST = MRKTAB(2,JMARK)
C
C     SAVE THE CURRENT PLOTTING POSITION AS THE REFERENCE POINT.
C
C
C     DRAW THE DATA MARK.
C
	ascii(imark)=ipnt
	isz=ipnt
	imin=100
	imax=-imin
	ipnt=ipnt+1
      DO 10 IPT=IPOINT,ILAST
      IB = TABT(IPT)
C
      IF(IB .EQ. 0)THEN
	sx(ipnt)=50
	sy(ipnt)=0
      ELSE
C
	movex = rshift(ib,3) - 3
      MOVEY = AND(IB,7) - 3
C
C     IF THE DATA MARK IS LARGE, DOUBLE THE DISPLACEMENT.
C
        IF(LARGE)THEN
          MOVEX = 2 * MOVEX
          MOVEY = 2 * MOVEY
        ENDIF
C
C     CALCULATE THE NEW POINT.
C
	if(movex.lt.imin)imin=movex
	if(movex.gt.imax)imax=movex
c
	iy=movey
	ix=movex
	iy=-iy
	if(iy.lt.0)iy=100+iy
	if(ix.lt.0)ix=100+ix
	sx(ipnt)=ix
	sy(ipnt)=iy
      ENDIF
	ipnt=ipnt+1
   10 CONTINUE
C
C
	if(imin.lt.0)imin=100+imin
	if(imax.lt.0)imax=100+imax
	sx(isz)=imin
	sy(isz)=imax
c
	sx(ipnt)=50
	sy(ipnt)=50
	ipnt=ipnt+1
100	continue
	ascii(89)=ipnt-1
	open(1,file='newfont')
	write(1,999)88,ibase,irast
999	format(1x,i2,2i4)
	write(1,998)(ascii(i),i=1,89)
998	format(1x,20i4)
	ilen=ascii(89)
	write(1,997)(sx(i),i=1,ilen)
	write(1,997)(sy(i),i=1,ilen)
997	format(1x,40i2)
	close(1)
C
      END
