c*
c* ef_get_arg_string.F
c*
c* Jonathan Callahan
c* Mar 17th 1999
c*
c*

c*
c* This returns the string which was passed as an argument
c* to the function.  
c*
c* (The maximum length of a string is CHARACTER*160 as 
c*  declared in xvariables.cmn.)

      SUBROUTINE EF_GET_ARG_STRING(id, iarg, text)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg, mr
      CHARACTER*(*) text

      INTEGER coded_pointer, uvar, item, start, end
      INTEGER mr_list(EF_MAX_ARGS)
 
c The memory resident list was cached by Ferret and must be retrieved
c to be accessible to the external function.
      CALL EF_GET_MR_LIST(mr_list)
      mr = mr_list(iarg)

c Here's how you check the variable type.
      IF (mr_category(mr) .NE. cat_string)
     .     WRITE (6,19) iarg
 19   FORMAT(' ERROR: non-string found for argument number ', I2)

c Here's how you trap the string, itself
        coded_pointer = mr_variable(mr)

c ... note that the uvar number and item number of the string are
c     packed together in the coded_pointer value
        IF ( coded_pointer .GT. 0 ) THEN
c ... as of V5.0 should always come here
           uvar = coded_pointer/1000
           item = coded_pointer - 1000*uvar
           start = uvar_item_start( item, uvar )
           end   = uvar_item_end  ( item, uvar )
c ... clip of the quote at start and end
           start = start + 1
           end = end - 1

           IF ( (end-start) .GT. 160 ) THEN
              write (6,29)
 29           FORMAT(' ERROR: string argument longer than 
     .             maximum [=160]')
           ELSE
              text = uvar_text(uvar)(start:end)
           ENDIF

        ELSE
c ... for future grids of strings
           STOP 'illegal string argument encoding'
!          pointer to array of strings encoded in memory(k,blk)
!          string = EXTRACT_FROM_STRING_GRID(memory(k,blk),i,j,k,l)
      ENDIF

      RETURN
      END

