	CHARACTER*13 FUNCTION TM_STRING ( dval )
*
*  Will take a double precision real and output it as a string. It will be
*  written as an integer if there is no fractional part and the value is
*  under 1,000,000 , otherwise it will be written out in G format.
*
*  written 3/87 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 03/15/87
* revision 0.10 - 02/15/89 - added temporary string
* revision 0.20 - 07/14/89 - changed format from 1PG12.6 to 1PG12.5 to catch
*			   - all real numbers
*
* argument
	REAL*8	dval
* local variable
	CHARACTER*13	temp

	temp(1:13) = ' '
	IF (ABS(dval) .LT. 1D6) THEN
	  IF (dval .EQ. DBLE(INT(dval))) THEN
	    WRITE (temp(1:12),'(I12)') INT(dval)
	    GOTO 999
	  ENDIF
	ENDIF

	WRITE (temp(1:12),'(1PG12.5)') dval

999	TM_STRING = temp
	RETURN
	END
