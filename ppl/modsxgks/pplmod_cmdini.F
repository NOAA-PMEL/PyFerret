	SUBROUTINE CMDINI
C** 
C**    @(#)cmdini.F	1.2    5/26/88
C**
C**
C***********************************************************************
C**
C**		    PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**	Subprogram called:
C**		CALL CMDINI
C**
C**	Parameters:
C**
C**	Description:
C**		Initializes the logical units and files necessary for
C**		both the command file and key file reads.
C**
C**	History:
C**	    Written: 
C**		26-JAN-86	by Donald W. Denbo
C**	    Modified:
C**		30-MAY-1986	by Donald W. Denbo
C**		  Changes made for VAX/VMS version
C**
C**
#ifdef unix
	INCLUDE 'pplv11inc/SYSTEM.INC'
	INCLUDE 'pplv11inc/SYMKEY.INC'
	INCLUDE 'pplv11inc/CMDCOM.INC'
	INCLUDE 'pplv11inc/LUNITS.INC'
#else
	INCLUDE 'TMAP_PPLV11INC:SYSTEM.INC'
	INCLUDE 'TMAP_PPLV11INC:SYMKEY.INC'
	INCLUDE 'TMAP_PPLV11INC:CMDCOM.INC'
	include 'TMAP_PPLV11INC:lunits.inc'
#endif
	CHARACTER FROM*80,TO*80,TEMP*255,LINE*255,CR*1,ESC*1
	INTEGER*4 STATUS,IFLAG

#ifdef STAR_1_SUPPORTED
	integer*1 BUF(80),CRB,ESCB
#else
	BYTE BUF(80),CRB,ESCB
#endif

	DATA CRB,ESCB/13,27/

*	Next two lines' variables init'd in PPLDATA.F *jd* 2.93
*	DATA ECHOF,DEBUGF,SMKEY2/.TRUE.,.FALSE.,.FALSE./
*	DATA KEYLN1,KEYLN2,ECHLUN,CMDLUN/21,22,23,24/

c	DATA FROM,TO/'PL5KEY:SYMBOL.KEY','SY:SYSTEM.KEY'/
	EQUIVALENCE (BUF(1),LINE),(CR,CRB),(ESC,ESCB)
C
#ifndef unix
	CALL ERRSET(36,.TRUE.,.FALSE.,.TRUE.,.FALSE.)
#endif
C
C	OPEN KEY FILE FOR SYMBOLS
C
C**	CALL COPY(FROM,TO,IER)
C**	IF(IER.NE.0)THEN
C**	    WRITE(LTTOUT,999)IER
C**999	    FORMAT(' Symbol key file copy error ',I3)
C**	    CALL EXIT
C**	ENDIF
#ifdef aix
*	OPEN(KEYLN1,FILE='PPL$KEY.001',STATUS='NEW')
#else
# ifdef NO_OPEN_RECORDTYPE
	OPEN(KEYLN1,FILE='PPL$KEY.001',STATUS='NEW')
# else
	OPEN(KEYLN1,FILE='PPL$KEY.001',STATUS='NEW',
     *  ORGANIZATION='INDEXED',ACCESS='KEYED',
     *  RECORDTYPE='VARIABLE',FORM='FORMATTED')
# endif
#endif
*	Following not allowed in f77 on DEC -- is it a don't care?? *jd* 2.93
*     *  KEY=(1:30:CHARACTER))
C
C	OPEN ECHO FILE
C
#ifdef NO_OPEN_CARRIAGECONTROL
	OPEN(ECHLUN,FILE='ECHO',STATUS='NEW')
#else
	OPEN(ECHLUN,FILE='ECHO',STATUS='NEW',CARRIAGECONTROL='LIST')
#endif

C
C	GET COMMAND LINE
C
c
c	rsx-11m
c
c	CALL GETMCR(BUF)
c	ICR=INDEX(LINE,CR)-1
c	IESC=INDEX(LINE,ESC)-1
c	ISTP=MAX0(ICR,IESC)
c	IS=INDEX(LINE,' ')+1
c	IF(IESC.GT.0)BATCHF=.TRUE.
c	IF(IS.GE.ISTP.OR.IS.EQ.1)THEN
c
c	end rsx-11m
c
c	vax/vms
c
#ifndef unix
	STAT=LIB$GET_FOREIGN(LINE,,ISTP)
#endif
	IS=1
	IF(LINE.EQ.' '.OR.ISTP.EQ.0)THEN
c
c	end vax/vms
c
C
C	NO FILE NAME READ
C
	    FROM='TT:'
	    LINE=' '
	    TERMF=.TRUE.
	ELSE
	    TEMP=LINE(IS:ISTP)
	    ISPC=INDEX(TEMP,' ')
	    IF(ISPC.GT.0)THEN
		FROM=TEMP(:ISPC-1)
		LINE=TEMP(ISPC+1:)
		IS=ISTP-IS+1-ISPC
	    ELSE
		FROM=TEMP
		LINE=' '
		IS=0
	    ENDIF
	    TEMP=FROM
	    itlen=lnblk(temp,255)
	    CALL WLDFLE('SY:.PPC',TEMP(1:itlen),FROM,STATUS,IFLAG)
	    TERMF=.FALSE.
	ENDIF
C
C	SET COMMAND READ
C
	CMDLEV=0
	CALL ATSTRT(FROM,LINE,IS,IER)
	RETURN
	END
