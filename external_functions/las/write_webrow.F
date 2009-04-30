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

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg
      CHARACTER*100 descrip


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      WRITE (descrip, 100)
      CALL ef_set_desc(id, descrip)
  100 FORMAT ('Write a webrowset file with cruise numbers and IDs')

      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance(id,  ABSTRACT,
     .     NORMAL, NORMAL, NORMAL)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'id')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'Cruise IDs (string)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'Cruise_Mask')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Mask =1 at start of each cruise')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      
      arg = 3
      CALL ef_set_arg_name(id, arg, 'filename')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'Filename to write')
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



      SUBROUTINE write_webrow_result_limits(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER nx

      nx = 1
      call ef_set_axis_limits(id, X_AXIS, 1, nx)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE write_webrow_compute(id, arg_1, arg_2, arg_3, result)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id, arg

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(2,mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      INTEGER slen, iunit, icruise, nd
      INTEGER i1,j1,k1,l1, i2,j2,k2,l2
      CHARACTER*512 fname
      CHARACTER*100 errtxt
      CHARACTER*512 buff

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

C  Get file name

      arg = 3
      CALL ef_get_arg_string(id, arg, fname)

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
      buff = "<webRowSet xmlns='http://java.sun.com/xml/ns/jdbc' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'"
      slen = 104
      WRITE (iunit, 1000) buff(1:slen)

      buff = "xsi:schemaLocation='http://java.sun.com/xml/ns/jdbc http://java.sun.com/xml/ns/jdbc/webrowset.xsd'>"
      slen = 99
      WRITE (iunit, 1000) buff(1:slen)

      WRITE (iunit, *) '<data>'

* Write cruise ids and numbers to the webrowset file...

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      k1 = arg_lo_ss(Z_AXIS,ARG1)
      l1 = arg_lo_ss(T_AXIS,ARG1)
      i = res_lo_ss(X_AXIS)

      icruise = 0
      DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
         IF (arg_2(i1,j1,k1,l1) .NE. bad_flag(ARG2)) THEN
            CALL EF_GET_STRING_ARG_ELEMENT(id, ARG1, arg_1,
     .                          i1,j1,k1,l1, slen, buff)
            icruise = icruise + 1 
	    IF (icruise .LT. 10) THEN
	       WRITE (iunit, 1010) icruise
	    ELSE IF (icruise .GE. 10) THEN
	       WRITE (iunit, 1012) icruise
	    ELSE IF (icruise .GE. 100) THEN
	       WRITE (iunit, 1014) icruise
	    ENDIF
            WRITE (iunit, 1020) buff(1:slen)
	 ENDIF
 100  CONTINUE

1010  FORMAT('<currentRow>'/'<columnValue>'/I1/'</columnValue>')
1012  FORMAT('<currentRow>'/'<columnValue>'/I2/'</columnValue>')
1014  FORMAT('<currentRow>'/'<columnValue>'/I3/'</columnValue>')
1020  FORMAT('<columnValue>'/A/'</columnValue>'/'</currentRow>')

* Write closing tag.
      WRITE (iunit, 1030) 
1030  FORMAT ('</data>'/'</webRowSet>')

      CLOSE (iunit, ERR = 5020 )
      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Y_AXIS)
      l = res_lo_ss(Y_AXIS)
      result(i,j,k,l) = 1.

      RETURN 

5010  CONTINUE
      WRITE(errtxt,*) 'Error opening file ', fname
      CALL EF_BAIL_OUT(id, errtxt)
      RETURN
5020  CONTINUE
      WRITE(errtxt,*) 'Error closing file ', fname
      CALL EF_BAIL_OUT(id, errtxt)
      RETURN

      END
