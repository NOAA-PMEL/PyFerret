      REAL*8 FUNCTION DATYPE(JT,NCHAR)
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
