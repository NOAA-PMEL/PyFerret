	SUBROUTINE SETFONT(fontdir,dirlen,dname,ifont,ii,kif,dnme,
     .               ifn,dnum,itype,dtype,iflen, dlen,ioff,
     .               icnt,ascii,base,rast,iadd,
     .		     sx,sy,id)

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
* Created 1.28.97 *kob*
*
*	Code in this routine taken from old pplmod_symbel.F routine.  This
*	was necessary to reorganize the symbel routine to remove an illegal
*	forward reference into a do loop.  This routine makes the 
*	pplmod_symbel a little cleaner.  This change has 
*	occurred in the middle of releasing v4.45 of 
*	Ferret to the world.  Aix, Sgi, HP, OSF and SunOs
*	have already been released, and this change is
*	not a part of them.
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. Remove commented-out code

        IMPLICIT NONE

	DIMENSION SX(12000),SY(12000),ASCII(127,4),BASE(4),RAST(4)
	DIMENSION DTYPE(35),DLEN(35),IOFF(4),IFLEN(4),ITYPE(4)
C
        REAL BASE, RAST
        INTEGER IFONT, II, IADD, ID
	LOGICAL SYMROT,READIR,PLOT,COLRF
c	CHARACTER PLTFIL*80
	CHARACTER ICOM*2,ICHR*1,STRING*120,DNME(35)*(*),DNUM(35)*(*)
	CHARACTER DNAME(35)*(*),KIF(4)*(*),IFN(4)*(*)
	CHARACTER*1 KSUB,KSUPER,KBKSP,KCALL,PENS(16)

* 1/03 *kob* - Add temp character buffer for g77 arbitrary text
*              concatenations
	CHARACTER*2048 temp_charbuff
	INTEGER*2 SX,SY

        integer       dirlen
        character*255 fontdir

	INTEGER*2 ICNT,IBASE,IRAST
*added for linux port
	INTEGER*4 IBASE4,IRAST4 
	INTEGER*2 ASCII
	INTEGER*2 DTYPE,DLEN,IOFF,IFLEN,ITYPE
	INTEGER COLOLD
	LOGICAL ASCFNT
        INTEGER IJ
	INTEGER TM_LENSTR1, tlen
C
C
	INCLUDE 'pltcom_dat.decl'
	INCLUDE 'PLTCOM.DAT'

	temp_charbuff = DNAME(id)
	OPEN(LUSY,FILE=fontdir(:dirlen)//'/'//temp_charbuff,STATUS='OLD',
     *	FORM='UNFORMATTED',ERR=5000)
C
C	SETUP FOR NEW FONT
C
	IFONT=II
	KIF(II)=DNME(ID)
	IFN(II)=DNUM(ID)
	ITYPE(II)=DTYPE(ID)
	IFLEN(II)=DLEN(ID)
C
C	SETUP IOFF
C
	IF(II.EQ.4)GOTO 1008
	DO 1007 IJ=II+1,4
	  IOFF(IJ)=12000
	  KIF(IJ)='@@'
	  IFN(IJ)='@@'
	  IFLEN(IJ)=0
 1007	CONTINUE	
	IOFF(II+1)=IOFF(II)+IFLEN(II)
C
C	READ NEW FONT
C
 1008	CONTINUE
	READ(LUSY,ERR=5010)ICNT,IBASE,IRAST
	IBASE4 = IBASE
	IRAST4 = IRAST
	READ(LUSY,ERR=5010)(ASCII(IJ,II),IJ=1,127)
	BASE(II)=FLOAT(IBASE4)
	RAST(II)=21./FLOAT(IRAST4)
	IADD=IOFF(II)+1
	READ(LUSY,ERR=5010)(SX(IJ),IJ=IADD,IADD+DLEN(ID)-1)
	READ(LUSY,ERR=5010)(SY(IJ),IJ=IADD,IADD+DLEN(ID)-1)
	CLOSE(LUSY)
	return

 5000	temp_charbuff = DNAME(ID)
	tlen = TM_LENSTR1(temp_charbuff)
	write(6,*) "setfont: Couldn't open font file:"//
     .               fontdir(:dirlen)//'/'//temp_charbuff(:tlen)
	stop
 5010	temp_charbuff = DNAME(ID)
	tlen = TM_LENSTR1(temp_charbuff)
	write(6,*) "setfont: Read error on font file:"//
     .               fontdir(:dirlen)//'/'//temp_charbuff(:tlen)
	stop
	end
