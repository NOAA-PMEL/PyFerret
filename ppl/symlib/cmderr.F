
	SUBROUTINE CMDERR(IER,LINE,ILEN,IPOS)
C   
C      @(#)cmderr.F	1.2    5/26/88
C  
C  
C                                                                      *
C  
C  		    PLOT+ Scientific Graphics System
C  
C                                                                      *
C  
C  
C  	Subprogram called:
C  		CALL CMDERR(IER,LINE,ILEN,IPOS)
C  
C  	Parameters:
C  		IER -- Error code
C  			0 = no error
C  			1 = symbol not found
C  			2 = unmatched single quote in parse
C  			3 = qualifier not found
C  			4 = then missing
C  			5 = nesting error for IF statement
C  			6 = logical operator not found
C  			7 = unmatched double quote in expression
C  			8 = nesting error for WHILE statement
C  			9 = syntax error
C  		       10 = unmatched (in LEV command
C  		       11 = exceeds maximum number of levels
C  		LINE -- Command line
C  		ILEN -- Length of LINE
C  		IPOS -- Position to mark
C  
C  	Description:
C  		Routine to print the input line and point to the
C  		offending element.  This command will abort the
C  		program if BATCHF is true.
C  
C  	History:
C  	    Written: 
C  		5-JAN-86	by Donald W. Denbo
C  	    Modified:
C  
C  
#ifdef unix
	INCLUDE 'cmdinc/SYSTEM.INC'
	INCLUDE 'cmdinc/CMDCOM.INC'
	INCLUDE 'pplinc/LUNITS.INC'
#else
	INCLUDE 'cmdinc:SYSTEM.INC'
	INCLUDE 'cmdinc:CMDCOM.INC'
	include 'pplinc:lunits.inc'
#endif
	CHARACTER LINE*(*),frmt*30
	CHARACTER ERRMES(11)*30
	DATA ERRMES/'Symbol not found','Unmatched ''',
     *	'Qualifier not recognized','THEN missing',
     *	'IF levels improperly nested','No operator found',
     *	'Unmatched "','WHILE levels improperly nested',
     *	'Syntax error','Unmatched (','Maximum no. levels exceeded'/
	IF(IER.EQ.0)RETURN
	WRITE(LTTOUT,999)' ',ERRMES(IER)
999	FORMAT(A1,1X,A30)
	WRITE(LTTOUT,998)' ',LINE(:ILEN)
998	FORMAT(A1,1X,A)
	IF(IPOS.LE.ILEN)THEN
	    WRITE(FRMT,997)IPOS
997	    FORMAT('(A1,',I3.3,'X,''^'')')
	    WRITE(LTTOUT,FRMT)' '
	ENDIF
	IF(ECHOF)THEN
	    WRITE(ECHLUN,999)'C',ERRMES(IER)
	    WRITE(ECHLUN,998)'C',LINE(:ILEN)
	    IF(IPOS.LE.ILEN)WRITE(ECHLUN,FRMT)'C'
	ENDIF
	IF(BATCHF)CALL EXIT
	IF(.NOT.TERMF)THEN
C
C	     If not interactive abort command file
C	     Return to lowest level
C
	    IF(.NOT.ECHOF)THEN
		WRITE(ECHLUN,999)'C',ERRMES(IER)
		WRITE(ECHLUN,998)'C',LINE(:ILEN)
		IF(IPOS.LE.ILEN)WRITE(ECHLUN,FRMT)'C'
	    ENDIF
	    CALL ATFLSH
	ENDIF
	RETURN
	END
