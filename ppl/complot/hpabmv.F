	SUBROUTINE HPABMV
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
C**    @(#)hpabmv.f	1.1    3/10/88
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
C	CALL HPABMV
C
C	THIS SUBROUTINE DRIVES THE HP7475 AND HP7470 PLOTTERS
C	THE CORRDINATES OF THE NEW POSITION ARE RECEIVED IN A AND B
C
C	D. DENBO   MARCH 1984
C
#ifdef unix
	include 'PLTCOM.DAT'
#else
	include 'pplinc:PLTCOM.DAT'
#endif
	IF(.NOT.PLTFLG)THEN
	CALL ZBUFFT('DF;SP1;',7)
	PLTFLG=.TRUE.
	LPEN=.NOT.PEN
	ENDIF
	IF(PEN.AND..NOT.LPEN)THEN
	CALL ZBUFFT('PD',2)
	ELSE IF(.NOT.PEN.AND.LPEN)THEN
	CALL ZBUFFT('PU',2)
	ELSE 
	CALL ZBUFFT(',',1)
	ENDIF
	DA=A-OLDHPA
	DB=B-OLDHPB
	IF(DA.EQ.0.AND.DB.EQ.0.)RETURN
	CALL HPOUT(INT(A))
	CALL ZBUFFT(',',1)
	CALL HPOUT(INT(B))
	OLDHPA=A
	OLDHPB=B
	RETURN
	END
#ifdef unix
	SUBROUTINE HPOUT(IA)
	character as*5
	DO 10 I=5,1,-1
	IA10=IA/10
	AS(I:I)=char(IA-IA10*10+48)
	IA=IA10
	IF(IA.EQ.0)GOTO 11
10	CONTINUE
	I=1
11	CALL ZBUFFT(AS(I:),6-I)
	RETURN
	END
#else
	SUBROUTINE HPOUT(IA)
	BYTE AS(5)
	DO 10 I=5,1,-1
	IA10=IA/10
	AS(I)=IA-IA10*10+48
	IA=IA10
	IF(IA.EQ.0)GOTO 11
10	CONTINUE
	I=1
11	CALL ZBUFFT(AS(I),6-I)
	RETURN
	END
#endif
