c*
c* ef_set_real_to_string.F
c*
c* Ansley Manke
c* June 2002
c*

c* This subroutine is called by an external function and formats a real
c* number as a string.  Calls internal Ferret routine tm_fmt.


* calling arguments:
* val		- REAL*4 value to be converted
* digits	- IF >0 maximum number of significant digits desired
*	        - ELSE  maximum number of decimal places desired
* max_len	- longest allowable length for string to be returned
* str_len	- (returned) actual length of the string generated
* outstring     - output string

      SUBROUTINE ef_set_real_to_string ( val, str_len, outstring )

      REAL val
      INTEGER digits, max_len, str_len
      CHARACTER*(*) outstring
      CHARACTER*8 TM_FMT

* internal declarations

      INTEGER digits, max_len
 
      digits = 8
      max_len = 8

      outstring = TM_FMT (val, digits, max_len, str_len)

      RETURN 
      END
