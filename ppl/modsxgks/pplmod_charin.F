	SUBROUTINE CHARIN(STR,IN,PRMT,INPL)
C** 
C**    @(#)charin.F	1.2    5/26/88
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
C**		CALL CHARIN(STR,IN,PRMP,INP)
C**
C**	Parameters:
C**		STR -- Output string (char*IN or byte(IN))
C**		IN -- Maximum input string length
C**		PRMP -- Prompt for terminal input
C**		INP -- Prompt lenght in characters
C**
C**	Description:
C**		CHARIN prompts the user for input from the terminal.
C**
C**	History:
C**	    Written: 
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**		30-MAY-1986	by Donald W. Denbo
C**		  Changes made for VAX/VMS version
C**
C**
C     Mod *JD* 3.5.91 problems with buffered terminal output in unix
*     Mod *jd* 8.30.93 to handle FORTRAN read error you get w XGKS if the 
*	window is clicked with the mouse.
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 


#ifdef unix
	include 'pplv11inc/CMDCOM.INC'
	include 'pplv11inc/LUNITS.INC'
#else
	include 'pplv11inc:CMDCOM.INC'
	include 'pplv11inc:lunits.inc'
#endif
	character str*(*),prmt*(*)
#ifdef unix
	write(LTTOUT,999)prmt(:inpl)
C     Mod format *JD* 3.5.91 get problem with output held in buffer
# ifdef FORTRAN_90
999	format(a)
# else
999	format(1x,a,$) !original
# endif
C OK in Sun?? 999	format(a)
	call flush(LTTOUT)
C Leave this out JD 3.5.91	call flush(LTTOUT)
100	read(CMDLUN,998,err=100,end=100)str(:in)
998	format(a)
#else
        INTEGER*2 IOSB(4)
        BYTE INB(255),PRMTB(30)
        INCLUDE '($IODEF)'
        STR=' '
        READ(PRMT,'(30A1)')PRMTB
        CALL SYS$ASSIGN('TT',CHAN,,)
        CALL SYS$QIOW(,%VAL(CHAN),%VAL(IO$_READPROMPT),IOSB,,,
     *  INB,%VAL(IN),,,PRMTB,%VAL(INP))
        CALL SYS$DASSGN(%VAL(CHAN))
        WRITE(STR,'(255A1)')(INB(I),I=1,IOSB(2))
#endif
	RETURN
	END
