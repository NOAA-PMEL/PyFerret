	SUBROUTINE NUMCDE_OFF(XZ,STR,NC,FRMTZ,DMS,OFF)
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
* revised for TMAP
* from PPLUS :
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]NUMCDE.FOR

C**
C**    from numcde.f
C 1/2006 acm Add an offset to the value before formatting it 
C V600: *acm* 5/06 Allow seconds in DMS axis labelling; had removed this.
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

c
	CHARACTER*(*) FRMTZ
	CHARACTER*20 FRMT,STR,TEMP,STR_DMS
        INTEGER TM_LENSTR1, dms, imin, isec, dot
        REAL frac, minutes, seconds, off
        REAL*8 val, x
C
	FRMT=FRMTZ
	TEMP=FRMTZ
	CALL UPPER(TEMP,20)
	X=XZ + off

	ILAT = INDEX (TEMP, ',''LAT'')')
	IF (ILAT .NE. 0) THEN
	  IF (X .LT. 0) THEN
	    X=-X
	    FRMT (ILAT+1:) = '''#S'')'
	  ELSE IF (X .EQ. 0.) THEN
	    FRMT (ILAT+1:) = '''#'')'
	  ELSE
	    FRMT (ILAT+1:) = '''#N'')'
	  ENDIF
	ELSE
	  ILON = INDEX (TEMP, ',''LON'')') + INDEX (TEMP, ',''LONW'')')
	  ILONE = INDEX (TEMP, ',''LONE'')')
	  IF (ILON .NE. 0) THEN
	    IF (X .GT. 180.) THEN
	      X=360.-X
	      FRMT (ILON+1:) = '''#E'')'
	    ELSE IF (X .EQ. 180.) THEN
	      FRMT (ILON+1:) = '''#'')'
	    ELSE IF (X .LT. 0.) THEN
	      FRMT (ILON+1:) = '''#E'')'
	      X = -X
	    ELSE
	      FRMT (ILON+1:) = '''#W'')'
	    ENDIF
	  ELSE IF (ILONE .NE. 0)THEN
* TMAP insert 2/13/89
	    X = MOD( X, 360. )
* ... end of TMAP insert
	    IF (X .GT.180.) THEN
	      X=360.-X
	      FRMT (ILONE+1:) = '''#W'')'
	    ELSE IF (X .EQ. 180.) THEN
	      FRMT (ILONE+1:) = '''#'')'
	    ELSE IF (X .LT. 0.) THEN
	      FRMT (ILONE+1:) = '''#W'')'
	      X = -X
	    ELSE IF (X .NE. 0.) THEN
	      FRMT (ILONE+1:) = '''#E'')'
            ELSE
	      FRMT (ILONE+1:) = '''#'')'
	    ENDIF
	  ENDIF
	ENDIF
C
	FLAG=0
	IF(FRMT(1:2).EQ.'(I'.OR.FRMT(1:2).EQ.'(i')THEN
            xr = x
	    WRITE(STR,FRMT)IFIX(XR+SIGN(.5,off))
	ELSE
	    WRITE(STR,FRMT)X
	END IF
C     STRIP OFF LEADING ZERO AND SET FLAG
	IF(FRMT(1:4).EQ.'(''0''')THEN
	    TEMP=STR(2:20)
	    STR=TEMP
	    FLAG=1
	END IF
C     LEFT JUSTIFY STRING
	NC=0
100	IF(STR(1:1).EQ.' ')THEN
	    TEMP=STR(2:20)
	    STR=TEMP
	    NC=NC+1
	    IF(NC.LE.20)GOTO 100
	END IF
C     ADD LEADING ZERO
	IF(FLAG.EQ.1.AND.ABS(X).LT.1.1)THEN
	    IF(STR(1:1).EQ.'.')THEN
		TEMP='0'
		TEMP(2:)=STR(1:19)
	    ENDIF
	    IF(STR(1:2).EQ.'+.'.OR.STR(1:2).EQ.'-.')THEN
		TEMP=STR(1:1)
		TEMP(2:2)='0'
		TEMP(3:)=STR(2:18)
	    ENDIF
	    STR=TEMP
	END IF
	NC=LNBLK(STR,20)

* Reformat the value into deg,min,sec if requested.
* This means moving the degree symbol after the degrees
* and putting the direction (N,S,E,W) at the end
* For longitudes larger than 180, format them as is and
* translate to west longitude afterwards.
 
        STR_DMS = ' '
        IF (dms .GE. 1) THEN
            
           dot = INDEX(str,'.')
           deg = INDEX(str,'#')
           IF (dot .EQ. 0) dot = TM_LENSTR1(str) +1
           val = xz + off
           IF (val .LT. 0) val = x

           ideg = INT(val)
           frac = val - FLOAT(ideg)
           minutes = 60.*frac
           imin = INT(minutes+0.1)  ! sun at least, turns 40 into 39 sometimes...
           frac = minutes - FLOAT(imin)
           seconds = 60.* frac
           isec = INT(seconds+0.1)

           IF (ideg .GT. 180.) THEN
              ideg = 360-ideg
              imin = 60-imin
           ENDIF

           IF (dms .EQ. 2) THEN  ! write as deg,min,sec
              IF (ideg .LT. 100) THEN
                 WRITE (STR_DMS, 1100) ideg, imin, isec,
     .               str(deg+1:deg+1)
              ELSE
                 WRITE (STR_DMS, 1101) ideg, imin, isec, 
     .               str(deg+1:deg+1)
              ENDIF
1100          FORMAT (I2.2, '#', I2.2, '''', I2.2, '''', '''', A1)
1101          FORMAT (I3.3, '#', I2.2, '''', I2.2, '''', '''', A1)

           ELSE IF (dms .EQ. 1) THEN  ! break into deg,min
              IF (ideg .LT. 100) THEN
                 WRITE (STR_DMS, 1200) ideg, imin, 
     .               str(deg+1:deg+1)
              ELSE
                 WRITE (STR_DMS, 1201) ideg, imin, 
     .               str(deg+1:deg+1)
              ENDIF
1200          FORMAT (I2.2, '#', I2.2, '''', A1)
1201          FORMAT (I3.3, '#', I2.2, '''', A1)

           ELSE IF (dms .EQ. 3) THEN  ! write as deg,min.frac

* get number of digits from the format given by the format that came into this routine.
* seeing if it is Fn.m.  If M=0, use minutes only. If M=1, then use one digit for the 
* fractional part of the mintues, otherwise write two digits. Round rather than truncating
* for degrees and fractions.

              dot = INDEX(frmtz,'.')
              READ (frmtz(dot+1:dot+1), *) digits
              IF (digits .EQ. 0) THEN   ! degrees, minutes only
                 imin = INT(minutes+0.5)
                 IF (ideg .LT. 100) THEN
                    WRITE (STR_DMS, 1300) ideg, imin,
     .               str(deg+1:deg+1)
                 ELSE
                    WRITE (STR_DMS, 1301) ideg, imin, 
     .               str(deg+1:deg+1)
                 ENDIF
1300             FORMAT (I2.2, '#', I2.2, '''', A1)
1301             FORMAT (I3.3, '#', I2.2, '''', A1)
              ELSE IF (digits .EQ. 1) THEN  ! degrees,min.f

                 IF (ideg .LT. 100) THEN
                    WRITE (STR_DMS, 1400) ideg, minutes, 
     .               str(deg+1:deg+1)
                 ELSE
                    WRITE (STR_DMS, 1401) ideg, minutes, 
     .               str(deg+1:deg+1)
                 ENDIF
1400             FORMAT (I2.2, '#', F4.1, '''', A1)
1401             FORMAT (I3.3, '#', F4.1, '''', A1)
              ELSE                          ! degrees,min.ff
                 IF (ideg .LT. 100) THEN
                    WRITE (STR_DMS, 1500) ideg, minutes, 
     .               str(deg+1:deg+1)
                 ELSE
                    WRITE (STR_DMS, 1501) ideg, minutes, 
     .               str(deg+1:deg+1)
                 ENDIF
1500             FORMAT (I2.2, '#', F5.2, '''', A1)
1501             FORMAT (I3.3, '#', F5.2, '''', A1)
              ENDIF

           ELSE 
              STR_DMS = str
           ENDIF
           str = STR_DMS
           NC=LNBLK(STR,20)

        ENDIF

	RETURN
	END
