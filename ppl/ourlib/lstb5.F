      SUBROUTINE LSTBBO(IUNIT,HEADR)
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
C**    @(#)lstb5.F	1.1    3/10/88
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
CC  DEFINITION OF INDEXES USED FOR BIBO FORMAT:
C
#ifdef unix
	INCLUDE 'pplinc/BIBO.COM'
#else
        INCLUDE 'pplinc:BIBO.COM'
#endif
C
      COMMON /TBLE/PTAB(8,4),DTTAB(8,29),FTTAB(8,9),WTTAB(8,11)
      COMMON /TBLE/STTAB(8,19),PCTAB(8,5),UNIT(4,15)
      COMMON /NTBLE/NPTAB,NDTTAB,NFTTAB,NWTTAB
      COMMON /NTBLE/NSTTAB,NPCTAB,NUNIT

	REAL LINE1,LINE2,BLNK
        DIMENSION HEADR(145),LINE1(11),LINE2(15)
#ifdef unix
	CHARACTER FDTE*24
#ifdef FORTRAN_90
	integer iblnk
	data iblnk/Z'20202020'/
	equivalence (blnk,iblnk)
#else
	data blnk/'    '/
#endif
	CALL FDATE(FDTE)
#else
	character dte*10,tme*10
	data blnk/'    '/
	call date(dte)
	call time(tme)
#endif

      CALL FILL(PTAB,NPTAB,HEADR(PROJ),LINE1,IER)
#ifdef unix
      WRITE(IUNIT,1000)(LINE1(I),I=1,6),FDTE
 1000 FORMAT(/1X,'PROJECT: ',6A4,T57,A24)
#else
      WRITE(IUNIT,1000)(LINE1(I),I=1,6),dte,tme
 1000 FORMAT(/1X,'PROJECT: ',6A4,T57,a10,3x,a10)
#endif
C        FIND THE END OF THE COMMENT FIELD:
      DO 200 I=1,5
      J1=(5-I)*20+HEAD
      J2=J1+19
      DO 100 J=J1,J2
      IF((HEADR(J).NE.BLNK).AND.(HEADR(J).NE.0.))GO TO 300
  100 CONTINUE
  200 CONTINUE
      GO TO 320

  300 WRITE(IUNIT,2000)(HEADR(K),K=HEAD,J2)
 2000 FORMAT(10(/1X,20A4))

  320 WRITE(IUNIT,3000)
 3000 FORMAT(1X,'GEOGRAPHIC INFORMATION:',T35,
     1   'DESCRIPTIVE INFORMATION:'/)

      JLEFT=1
      JRITE=1

  400 CALL COL1(HEADR,JLEFT,LINE1)
      CALL COL2(HEADR,JRITE,LINE2)
      WRITE(IUNIT,5000)(LINE1(I),I=1,9),(LINE2(I),I=1,12)
 5000 FORMAT(1X,8A4,A1,11A4,A3)
      JLEFT=JLEFT+1
      JRITE=JRITE+1
      IF(JRITE.LE.16)GO TO 400

      WRITE(IUNIT,6000)
 6000 FORMAT(1X)
      RETURN
      END
C*F45V1P0*

      SUBROUTINE COL1(HEADR,JLEFT,LINE1)

      DIMENSION HEADR(145),LINE1(11)

      CHARACTER TE*50,NS(2)*1,EW(2)*1
	REAL LINE1,BLNK
C
CC  DEFINITION OF INDEXES USED FOR BIBO FORMAT:
C
#ifdef unix
	INCLUDE 'pplinc/BIBO.COM'
#else
        INCLUDE 'pplinc:BIBO.COM'
#endif


#ifdef FORTRAN_90
	integer iblnk
	data iblnk/Z'20202020'/
	equivalence (blnk,iblnk)
#else
	data blnk/'    '/
#endif

      DATA NS/'N','S'/,EW/'E','W'/
      IF(JLEFT.LE.7)GO TO 100
      DO 50 I=1,11
   50 LINE1(I)=BLNK
      RETURN

  100 GO TO (1,2,3,4,5,6,7),JLEFT

1     WRITE(TE,1000)HEADR(MOOR),HEADR(MOOR+1)
 1000 FORMAT('STATION ID:',T13,2A4)
      GOTO 200

2     WRITE(TE,2000)HEADR(INST),HEADR(INST+1)
 2000 FORMAT('INSTRUMENT:',T13,2A4)
      GOTO 200

    3 ILAT=HEADR(SLAT)/100.0
	IF(HEADR(SLAT).GE.0.)THEN
	INS=1
	ELSE
	INS=2
	ENDIF
      RLAT=ABS(HEADR(SLAT)-ILAT*100.0)
      WRITE(TE,3000)IABS(ILAT),RLAT,NS(INS)
 3000 FORMAT('LATITUDE:',2X,I4,1X,F4.1,1X,A1)
      GOTO 200

    4 ILONG=HEADR(SLON)/100.0
	IF(HEADR(SLON).GE.0.)THEN
	IEW=1
	ELSE
	IEW=2
	ENDIF
      RLONG=ABS(HEADR(SLON)-ILONG*100.0)
      WRITE(TE,4000)IABS(ILONG),RLONG,EW(IEW)
 4000 FORMAT('LONGITUDE:',1X,I4,1X,F4.1,1X,A1)
      GOTO 200

5     WRITE(TE,5000)HEADR(SDEP)
 5000 FORMAT('STA DEPTH:',T13,F8.2,' (METERS)')
      GOTO 200

6     WRITE(TE,6000)HEADR(INDEP)
 6000 FORMAT('NOM DEPTH:',T13,F8.2,' (METERS)')
      GOTO 200

7     WRITE(TE,7000)HEADR(IADEP)
 7000 FORMAT('ACT DEPTH:',T13,F8.2,' (METERS)')
200   READ(TE,8000)LINE1
8000	FORMAT(11A4)
      RETURN
      END
C*F45V1P0*
      SUBROUTINE COL2(HEADR,JRITE,LINE2)
      DIMENSION HEADR(145),LINE2(15),LSTRNG(11)
      CHARACTER TE*60,STR*20
	REAL LINE2,LSTRNG
C
CC  DEFINITION OF INDEXES USED FOR BIBO FORMAT:
C
#ifdef unix
	INCLUDE 'pplinc/BIBO.COM'
#else
        INCLUDE 'pplinc:BIBO.COM'
#endif
      COMMON /TBLE/PTAB(8,4),DTTAB(8,29),FTTAB(8,9),WTTAB(8,11)
      COMMON /TBLE/STTAB(8,19),PCTAB(8,5),UNIT(4,15)
      COMMON /NTBLE/NPTAB,NDTTAB,NFTTAB,NWTTAB
      COMMON /NTBLE/NSTTAB,NPCTAB,NUNIT


      GO TO (1,2,3,4,5,6,61,7,8,9,10,11,12,13,14,15),JRITE

    1 CALL FILL(DTTAB,NDTTAB,HEADR(DTYPE),LSTRNG,IER)
      WRITE(TE,1000)(LSTRNG(I),I=1,10)
 1000 FORMAT('DATA:',T7,6A4,1X,4A4)
      GOTO 200

    2 IF(HEADR(SCODE).GT.0.0)GO TO 23
      JRITE=JRITE+1
      GO TO 3
   23 CALL FILL(STTAB,NSTTAB,HEADR(SCODE),LSTRNG,IER)
	WRITE(TE,3000)(LSTRNG(I),I=1,10)
	GOTO 200

    3 IF(HEADR(PCODE).GT.0.0)GO TO 33
      JRITE=JRITE+1
      GO TO 4
   33 CALL FILL(PCTAB,NPCTAB,HEADR(PCODE),LSTRNG,IER)
      WRITE(TE,3000)(LSTRNG(I),I=1,10)
 3000 FORMAT(T7,6A4,1X,4A4)
      GOTO 200

    4 IF(HEADR(MODEN).GT.0.0)GO TO 43
      JRITE=JRITE+1
      GO TO 5
43    WRITE(TE,4000)IFIX(HEADR(MODEN))
 4000 FORMAT('MODE NUMBER:',T21,I10)
      GOTO 200

    5 WRITE(TE,4500)IFIX(HEADR(NPTS))
 4500 FORMAT('NUMBER OF POINTS:',T21,I10)
      GOTO 200

    6 CALL MTDAT(HEADR(STIME),STR)
	WRITE(TE,4800)STR
 4800 FORMAT('DATA START TIME:',T20,A18,'(GMT)')
      GOTO 200

61	IF(HEADR(SPTIME).EQ.0.0)THEN
	JRITE=JRITE+1
	GOTO 7
	ELSE
	CALL MTDAT(HEADR(SPTIME),STR)
	WRITE(TE,4900)STR
4900	FORMAT('DATA STOP TIME:',T20,A18,'(GMT)')
	ENDIF
	GOTO 200

    7 WRITE(TE,5000)HEADR(SRATE)
 5000 FORMAT('SAMPLING RATE:',T21,F10.2,' (MINUTES)')
      GOTO 200

    8 CALL FILL(FTTAB,NFTTAB,HEADR(FILT),LSTRNG,IER)
      WRITE(TE,5500)(LSTRNG(I),I=1,6)
 5500 FORMAT('FILTER:',T7,6A4)
      IF(HEADR(FILT).LE.0.0)JRITE=JRITE+2
      GOTO 200

    9 WRITE(TE,5800)HEADR(CUTLO)
 5800 FORMAT('LOW CUTOFF:',T21,1PE10.2,' (CPH)')
      GOTO 200

   10 WRITE(TE,5900)HEADR(CUTHI)
 5900 FORMAT('HIGH CUTOFF:',T21,1PE10.2,' (CPH)')
      GOTO 200

   11 IF(HEADR(WIND).GT.0.0)GO TO 113
      JRITE=JRITE+2
      GO TO 13
  113 CALL FILL(WTTAB,NWTTAB,HEADR(WIND),LSTRNG,IER)
      WRITE(TE,6000)(LSTRNG(I),I=1,6)
 6000 FORMAT('WINDOW:',T7,6A4)
      GOTO 200

   12 WRITE(TE,6300)HEADR(SIG)
 6300 FORMAT('WINDOW FALLOFF:',T21,1PE10.2,' (CPH)')
      GOTO 200

   13 WRITE(TE,6500)HEADR(MIN)
 6500 FORMAT('MINIMUM:',T21,1PE10.2)
      GOTO 200

   14 WRITE(TE,6800)HEADR(MAX)
 6800 FORMAT('MAXIMUM:',T21,1PE10.2)
      GOTO 200

   15 WRITE(TE,7000)HEADR(MEAN)
 7000 FORMAT('MEAN:',T21,1PE10.2)

200   READ(TE,8000)LINE2
8000  FORMAT(15A4)
      RETURN
      END
C*F45V1P0*
      SUBROUTINE FILL(TABLE,N,VALUE,STRING,IER)

      COMMON /TBLE/PTAB(8,4),DTTAB(8,29),FTTAB(8,9),WTTAB(8,11)
      COMMON /TBLE/STTAB(8,19),PCTAB(8,5),UNIT(4,15)
      COMMON /NTBLE/NPTAB,NDTTAB,NFTTAB,NWTTAB
      COMMON /NTBLE/NSTTAB,NPCTAB,NUNIT

      DIMENSION TABLE(8,N),STRING(11)
	character str*44
	real blnk
#ifdef FORTRAN_90
	integer iblnk
	data iblnk/Z'20202020'/
	equivalence (blnk,iblnk)
#else
	data blnk/'    '/
#endif

      IER=0
      DO 100 I=1,N
      IF(ABS(TABLE(1,I)-VALUE).LT. 0.01)GO TO 200
  100 CONTINUE

      WRITE(STR,1000)IFIX(VALUE)
 1000 FORMAT(14X,I10)
	read(str,1223)STRING
1223	format(11a4)
      IER=1
      K=1
      GO TO 400

  200 DO 300 J=1,6
  300 STRING(J)=TABLE(J+2,I)
      K=TABLE(2,I)+1

  400 DO 500 J=7,10
  500 STRING(J)=UNIT(J-6,K)
      STRING(11)=BLNK
      RETURN
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
