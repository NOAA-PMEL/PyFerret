      SUBROUTINE CMRDST(LUN,FILE,CH) 
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
C**    @(#)cmrdst.F	1.1    3/10/88
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
C     WRITTEN IN JULY 1980  THIS ROUTINE 
C     ALLOWS ALTERNAME INPUT  LUN,NAME AND 
C     TERMINATOR CHARACTER LT 
C 
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

      COMMON/CDFTN5/LN 
      COMMON/CEFTN5/IN,IA,LT,NAME,INTEMP
      COMMON/CPFTN5/PRMPT
      CHARACTER FILE*(*),CH*(*),IN*80,IA*1,LT*1,NAME*81,INTEMP*80
      CHARACTER PRMPT*40

#ifdef unix
      external init_cdepftn5
#else
      DATA LN,LT,NAME/5,'|','TT:'/,PRMPT/'('' *'',$)'/
#endif

      LN=LUN 
      IF(CH.NE.' ')LT=CH 
      IF(FILE(1:1).EQ.' ')RETURN 
      NAME=FILE 
      CLOSE(LN) 
#ifdef unix
	IF(NAME.EQ.'/dev/tty')THEN
#else
	if(name.eq.'TT:')then
#endif
	OPEN(LN,FILE=NAME,STATUS='UNKNOWN') 
	ELSE
#ifdef unix
	OPEN(LN,FILE=NAME,STATUS='OLD',ERR=1000)
#else
	OPEN(LN,FILE=NAME,STATUS='OLD',readonly,ERR=1000)
#endif
	ENDIF
      RETURN 
1000	WRITE(5,*)' ERROR OPENING: ',NAME
	STOP
      END 


#ifdef unix

CC    Created this for FORTRAN_90 compile, but will be acceptable by 
CC    all UNIX

      block data init_cdepftn5

      COMMON/CDFTN5/LN 
      COMMON/CEFTN5/IN,IA,LT,NAME,INTEMP
      COMMON/CPFTN5/PRMPT

      CHARACTER FILE*81,CH*1,IN*80,IA*1,LT*1,NAME*81,INTEMP*80
      CHARACTER PRMPT*40

      DATA LN,LT,NAME/5,'|','/dev/tty'/,PRMPT/'('' *'',$)'/
      DATA IN/' '/ 

      end

#endif



