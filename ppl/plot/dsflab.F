	SUBROUTINE DSFLAB
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
C**    @(#)dsflab.F	1.1    3/10/88
C** v552 *acm* 3/03 Longer strings: changes for multi-line labels.
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C	DSFLAB fills the PLOT5 labels with information from the
C	BIBO header of a DSF file.
C
	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'CMRD.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'BIBO.COM'
	INCLUDE 'TBLE.INC'
C
	REAL STR(11)
	CHARACTER MORINT*16,LAT*11,LON*11,NS*1,EW*1,DEP*10
	INTEGER MORLEN,LATLEN,LONLEN,DEPLEN
C	INCLUDE 'BIBO.DAT'
C
C	fill the XLAB 
C
	XLAB = 'Time (BIBO)'
C
C	fill the YLAB
C
	CALL FILL(DTTAB,NDTTAB,P(DTYPE),STR,IER)
	WRITE(YLAB,'(6A4,1X,4A4)')(STR(I),I=1,10)
	CALL LJUST(YLAB,2048,J)
	WRITE(MORINT,'(4A4)')P(MOOR),P(MOOR+1),P(INST),P(INST+1)
	CALL SQUISH(MORINT,1,16)
	MORLEN=LNBLK(MORINT,16)
	ILAT=P(SLAT)/100.0
	IF(P(SLAT).GE.0)THEN
	    NS='N'
	ELSE
	    NS='S'
	ENDIF
	RLAT=ABS(P(SLAT)-ILAT*100.0)
	IF(RLAT.EQ.0.)THEN
	    WRITE(LAT,999)IABS(ILAT),NS
999	    FORMAT(I4,'#',A1)
	ELSE
	    WRITE(LAT,998)IABS(ILAT),RLAT,NS
998	    FORMAT(I4,1X,F4.1,'#',A1)
	ENDIF
	CALL LJUST(LAT,11,LATLEN)
	ILON=P(SLON)/100.0
	IF(P(SLON).GE.0)THEN
	    EW='E'
	ELSE
	    EW='W'
	ENDIF
	RLON=ABS(P(SLON)-ILON*100.0)
	IF(RLON.EQ.0.)THEN
	    WRITE(LON,999)IABS(ILON),EW
	ELSE
	    WRITE(LON,998)IABS(ILON),RLON,EW
	ENDIF
	CALL LJUST(LON,11,LONLEN)
	WRITE(DEP,'(F10.1)')P(INDEP)
	CALL LJUST(DEP,10,DEPLEN)
C
C	create main label
C
	LAB1=MORINT(1:MORLEN)//'  '//LAT(1:LATLEN)//', '//
     *	    LON(1:LONLEN)//'  ('//DEP(1:DEPLEN)//' M)'
	RETURN
	END
	SUBROUTINE LJUST(STR,INLEN,OUTLEN)
C
C	LJUST left justifies a string and returns its length
C
	INTEGER INLEN,OUTLEN

#ifdef FORTRAN_90
	character str*(*)
#else
	CHARACTER STR*2048
#endif

	IF(STR.EQ.' ')THEN
	    OUTLEN=1
	    RETURN
	ENDIF
100	IF(STR(1:1).NE.' ')GOTO 200
	STR(1:INLEN)=STR(2:INLEN)
	GOTO 100
200	OUTLEN=LNBLK(STR,INLEN)
	RETURN
	END
