	SUBROUTINE CLSPPL
C** 
C**    @(#)clsppl.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
#ifdef unix
	INCLUDE 'cmdinc/SYMKEY.INC'
	INCLUDE 'cmdinc/CMDCOM.INC'
	INCLUDE 'pplinc/LUNITS.INC'
	INCLUDE 'pltinc/PLTCOM.DAT'
#else
	INCLUDE 'cmdinc:SYMKEY.INC'
	INCLUDE 'cmdinc:CMDCOM.INC'
	INCLUDE 'pplinc:LUNITS.INC'
	INCLUDE 'pltinc:PLTCOM.DAT'
#endif
#ifdef core
	include 'pltinc/CORECM.INC'
#endif
C
C	CLOSES FILES USED BY PLOT5+
C
	CALL ATFLSH
#ifdef unix
	call dbmclose(keyln1)
	call unlink('PPL$KEY.001.pag')
	call unlink('PPL$KEY.001.dir')
#else
	close(keyln1,disp='delete')
#endif
#ifdef core
	if(gksopn)then
	    call closeretainseg()
	    call deselectvwsurf(vsurf)
	    call terminatecore()
	endif
#endif
	IF(PLTFLG)THEN
	    PEN = .FALSE.
	    CALL ZABMV
	    CALL BINFSH
	    PLTFLG = .FALSE.
	ENDIF
	RETURN
	END
