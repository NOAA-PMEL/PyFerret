c This will be in fer/ef_utility.  For now, compile as part of
c the examples/ external functions for faster re-compiling (and while
c stout is offline for its upgrade; some libraries I need to compile
c Ferret even under solaris seem to be on stout
c*

c*
c* ef_get_string_arg_element.F
c*
c*  acm 12/03  from ef_get_arg_string
c*
c*
c* This returns an element of the string array which was passed as an argument
c* to the function.  


      SUBROUTINE ef_get_string_arg_element(id, iarg, arg, i,j,k,l, text)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg, slen, i, j, k, l
      CHARACTER*(*) text
      REAL arg(*)

      INTEGER max_line_len, mr, lox, hix, loy, hiy, loz, hiz, lot, hit
      INTEGER mr_list(EF_MAX_ARGS)
      CHARACTER buff*32
      PARAMETER (max_line_len = 512)
      
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
 
c The memory resident list was cached by Ferret and must be retrieved
c to be accessible to the external function.
      CALL EF_GET_MR_LIST(mr_list)
      mr = mr_list(iarg)
      
      IF ( mr .EQ. 0 ) THEN
* we must have been called during custom axis creation time. Use
* ef_get_one_arg_string for this.
          buff = 
     .   'Use ef_get_one_arg_string to get one string during init phase'
          GOTO 5000

      ELSE

* Here's how you check the variable type.
* (this should never happen as checks against it are made previously) 
        IF (mr_type(mr) .NE. ptype_string) THEN
           WRITE (buff,4000) iarg
 4000      FORMAT ('Argument',I3,' is not a string')
           GOTO 5000
        ENDIF

        lox = arg_lo_ss(X_AXIS,iarg)
        hix = arg_hi_ss(X_AXIS,iarg)
        loy = arg_lo_ss(Y_AXIS,iarg)
        hiy = arg_hi_ss(Y_AXIS,iarg)
        loz = arg_lo_ss(Z_AXIS,iarg)
        hiz = arg_hi_ss(Z_AXIS,iarg)
        lot = arg_lo_ss(T_AXIS,iarg)
        hit = arg_hi_ss(T_AXIS,iarg)

* Inside ef_get_string_arg_element_sub, arg is a real*8 array of pointers, and
* we can call GET_STRING_ELEMENT to get a string element.

        CALL ef_get_string_arg_element_sub (lox, hix, loy, hiy,
     .                              loz, hiz, lot, hit,
     .			            arg, i, j, k, l, max_line_len, 
     .			            slen, text )

      ENDIF

      RETURN

* return an error message     
 5000 CALL EF_BAIL_OUT(id, buff)

      END
