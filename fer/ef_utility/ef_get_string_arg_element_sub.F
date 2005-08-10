c This will be in fer/ef_utility.  For now, compile as part of
* the examples/ external functions for faster re-compiling (and while
* stout is offline for its upgrade; some libraries I need to compile
* Ferret even under solaris seem to be on stout
*
* ef_get_string_arg_element_sub.F
*
*  acm 12/03  from ef_get_arg_string
*
* This returns an element of the string array which was passed as an argument
* to the function.  

* Inside this subroutine arg is a 4-D real*8 array of pointers, and
* we can call GET_STRING_ELEMENT to get a string element.

      SUBROUTINE ef_get_string_arg_element_sub (lox, hix, loy, hiy,
     .                     loz, hiz, lot, hit, arg, i, j, k, l, 
     .			   max_line_len, slen, text )

      INTEGER i, j, k, l, lox, hix, loy, hiy, loz, hiz, lot, hit, 
     .  slen, max_line_len
      CHARACTER*(*) text
      REAL*8 arg(lox:hix, loy:hiy, loz:hiz, lot:hit)

      CHARACTER	GET_STRING_ELEMENT*512
    

      text = GET_STRING_ELEMENT (lox, hix, loy, hiy, loz, hiz, lot, hit,
     .			         arg, i, j, k, l, max_line_len, slen)

      RETURN

      END
