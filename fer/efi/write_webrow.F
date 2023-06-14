*
* write_webrow.F
*
* Ansley Manke
* April 2009
*
* This function writes a special webrowset file for LAS
*
*
* In this subroutine we provide information about
* the function.  The user configurable information 
* consists of the following:
*
* descr              Text description of the function
*
* num_args           Required number of arguments
*
* axis_inheritance   Type of axis for the result
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )
* 
*
* For each argument we provide the following information:
*
* name               Text name for an argument
*
* unit               Text units for an argument
*
* desc               Text description of an argument
*
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE write_webrow_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'Write a webrowset file with cruise numbers and IDs')
      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT, NORMAL,
     .                                NORMAL,   NORMAL,
     .                                NORMAL,   NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'id')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'Cruise IDs (string) on X axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'name')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'Cruise name (string) on X axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'Cruise_Mask')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,
     .                'Mask on X axis; =1 at start of each cruise')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'filename')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'Filename to write')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


      SUBROUTINE write_webrow_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_axis_limits(id, X_AXIS, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE write_webrow_compute(id, arg_1, arg_2, arg_3, arg_4,
     .                                result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .           mem1loz:mem1hiz, mem1lot:mem1hit,
     .           mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit,
     .           mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy,
     .           mem3loz:mem3hiz, mem3lot:mem3hit,
     .           mem3loe:mem3hie, mem3lof:mem3hif)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy,
     .           mem4loz:mem4hiz, mem4lot:mem4hit,
     .           mem4loe:mem4hie, mem4lof:mem4hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l, m, n
      INTEGER slen, blen, iunit, icruise, nd
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      CHARACTER*512 fname
      CHARACTER*100 errtxt
      CHARACTER*512 buff, buff_cname

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

C  Get file name

      CALL ef_get_arg_string(id, ARG4, fname)

      iunit = 19

       OPEN( UNIT        = iunit,
     .       FILE        = fname,
     .       FORM        = 'FORMATTED',
     .       ACCESS      = 'SEQUENTIAL',
     .       STATUS      = 'REPLACE',
     .       ERR         = 5010 )

* Write webrowset header.

      buff = "<?xml version='1.0'?>"
      slen = 21
      WRITE (iunit, 1000) buff(1:slen)
 1000 FORMAT(A)

c      buff = "<webRowSet xmlns='http://java.sun.com/xml/ns/jdbc' " //
c     .       "xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'"
c      slen = 104
c      WRITE (iunit, 1000) buff(1:slen)
c
c      buff = "xsi:schemaLocation='http://java.sun.com/xml/ns/jdbc " //
c     .       "http://java.sun.com/xml/ns/jdbc/webrowset.xsd'>"
c      slen = 99
c      WRITE (iunit, 1000) buff(1:slen)
      
      buff = "<webRowSet>"
      slen = 99
      WRITE (iunit, 1000) buff(1:slen)

      WRITE (iunit, *) '<data>'

* Write cruise ids and numbers to the webrowset file...

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      k1 = arg_lo_ss(Z_AXIS,ARG1)
      l1 = arg_lo_ss(T_AXIS,ARG1)
      m1 = arg_lo_ss(E_AXIS,ARG1)
      n1 = arg_lo_ss(F_AXIS,ARG1)

* If the cruise_name is different from the cruise_id,
* write the cruise_name in parentheses after the ID.

      icruise = 0
      DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
         IF ( arg_3(i1,j1,k1,l1,m1,n1) .NE. bad_flag(ARG3) ) THEN
            icruise = icruise + 1 
            IF ( icruise .LT. 10 ) THEN
               WRITE (iunit, 1010) icruise
            ELSE IF ( icruise .LT. 100 ) THEN
               WRITE (iunit, 1020) icruise
            ELSE IF ( icruise .LT. 1000 ) THEN
               WRITE (iunit, 1030) icruise
            ELSE
               WRITE (iunit, 1040) icruise
            ENDIF

 1010       FORMAT('<currentRow>'/'<columnValue>'/I1/'</columnValue>')
 1020       FORMAT('<currentRow>'/'<columnValue>'/I2/'</columnValue>')
 1030       FORMAT('<currentRow>'/'<columnValue>'/I3/'</columnValue>')
 1040       FORMAT('<currentRow>'/'<columnValue>'/I4/'</columnValue>')

            CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG1, arg_1,
     .                         i1, j1, k1, l1, m1, n1, slen, buff)
            CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG2, arg_2,
     .                         i1, j1, k1, l1, m1, n1, blen, buff_cname)
            IF ( slen .EQ. blen ) THEN
               IF ( buff(:slen) .EQ. buff_cname(:slen) ) blen = 0
            ENDIF
            IF ( blen .GT. 0 ) THEN
               buff(slen+1:slen+2) = ' ('
               slen = slen + 2
               buff(slen+1:slen+blen) = buff_cname(:blen)
               slen = slen + blen
               buff(slen+1:slen+1) = ')'
               slen = slen + 1
            ENDIF

            WRITE (iunit, 1090) buff(1:slen)
 1090       FORMAT('<columnValue>'/A/'</columnValue>'/'</currentRow>')
         ENDIF
  100 CONTINUE


* Write closing tag.
      WRITE (iunit, 1110) 
 1110 FORMAT ('</data>'/'</webRowSet>')

      CLOSE (iunit, ERR = 5020)
      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)
      result(i,j,k,l,m,n) = 1.

      RETURN 

 5010 CONTINUE
      WRITE(errtxt,*) 'Error opening file ', fname
      CALL EF_BAIL_OUT(id, errtxt)
      RETURN

 5020 CONTINUE
      WRITE(errtxt,*) 'Error closing file ', fname
      CALL EF_BAIL_OUT(id, errtxt)
      RETURN

      END
