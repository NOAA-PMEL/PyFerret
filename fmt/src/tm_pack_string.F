	SUBROUTINE TM_PACK_STRING(in_string,out_string,start,end,len_out)

* remove blanks and tabs from the designated portion of the string

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 03/04/87
*
* calling argument declarations:
	INTEGER		start, end, len_out
	CHARACTER*(*)	in_string, out_string

* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9) )

* internal variable declarations:
	INTEGER	i_in, i_out, str_len, str_end

* true string length
	str_len = MIN( LEN(in_string), LEN(out_string) )
	str_end = MIN( str_len, end )

* transfer all characters until "start" to output string
	DO 10 i_in = 1, start
 10	out_string( i_in:i_in ) = in_string( i_in:i_in )

* pack the indicated region
	i_out = start

	DO 100 i_in = start, str_end
* ... skip hole ?
	   IF ( in_string( i_in:i_in ) .EQ. ' '
     .	   .OR. in_string( i_in:i_in ) .EQ. tab ) GOTO 100

* ... transfer character to output
	   out_string( i_out:i_out ) = in_string( i_in:i_in )
	   i_out = i_out + 1
 100	CONTINUE
	   
* transfer the remainder to the output
	DO 200 i_in = str_end+1, str_len
	   out_string( i_out:i_out ) = in_string( i_in:i_in )
	   i_out = i_out + 1
 200	CONTINUE

* length of packed output string
	len_out = i_out - 1

* blank out old shifted characters at right hand end
	DO 300 i_out = i_out, str_len
 300	out_string(i_out:i_out) = ' '

	RETURN
	END
