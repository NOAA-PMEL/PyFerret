      REAL*8 FUNCTION DATYPE(JT,NCHAR)
C** 
C**    @(#)datype5.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
      DIMENSION LEN(48)
	character type(48)*8,ty*8
      DATA TYPE/'MAJ CRNT','MIN CRNT','ALG CRNT','CRS CRNT',
     *     'MAJ WIND','MIN WIND','MAJ STRS','MIN STRS',
     *     'ATM PRES','WET TEMP','DRY TEMP',
     1     'WIND SPD','WIND DIR',' ',
     1     'AIR U VL','AIR V VL','TAU X','TAU Y',
     2     'SEA LVL','AIR TEMP','RTR SPD','VEC SPD','VANE DIR',
     3     'TEMP','U VEL','V VEL','PRESSURE','TILT','EDDY KE',
     4     'EDDY PE','SALNIITY','COND','ISO DPTH','SIGMA-T','X','Y',
     5	   'MODE AMP','ALNG WND','CRS WND','ALG STRS','CRS STRS',
     6     'WND STRD','WND STRC','AJD SLVL','ZONAL HF','PLOAR HF',
     7     'HEAT CNT','DYN HGT'/
      DATA LEN/8,8,8,8,8,8,8,8,8,8,8,8,8,1,8,8,5,5,7,8,7,7,8,4,5,
     *     5,8,4,7,7,8,4,8,7,1,1,8,8,7,8,8,8,8,8,8,8,8,7/

      IF(JT.GE.12 .AND. JT.LE.59)GO TO 100
      write(ty,1000)JT
 1000 FORMAT(I3)
	read(ty,999)datype
999	format(a8)
	NCHAR=3
      GO TO 200

  100 read(type(jt-11),999)datype
      NCHAR=LEN(JT-11)
      IF(NCHAR.LE.0)NCHAR=1
 200  RETURN
      END
