      SUBROUTINE CMRDST(LUN,FILE,CH) 
C** 
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



