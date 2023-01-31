*
* list_value_xml.F
*
* Ansley Manke
* May 24, 2011
* V691+*acm* 8/14  Fix ticket 2190: long strings were being cut off.
*
* This function writes a string value to an xml string, and with a given tag 
* and option to enclose it in CDATA escape. e.g.
*
*		<label><![CDATA[Z (m):]]></label>
*		<value><![CDATA[0]]></value>
*
* it then lists the string to the named file.
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


      SUBROUTINE list_value_xml_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*120 fcn_descr

      WRITE (fcn_descr,100) 
      CALL ef_set_desc(id, fcn_descr)
  100 FORMAT('write a string in xml format with specified ',
     .       'tag and optional CDATA escape sequence')

      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT, NORMAL,
     .                                NORMAL,   NORMAL,
     .                                NORMAL,   NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'tag')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'tag to open and close with')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'value')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'single string value to write')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'cdata_flag')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 
     .'enter 1 to request CDATA escape, 0 otherwise')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'outputfile')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'the filename to write to')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE list_value_xml_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER arg, my_lo, my_hi

      arg = 1
      my_lo = 1
      my_hi = 1
      CALL ef_set_axis_limits(id, X_AXIS, my_lo, my_hi)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE list_value_xml_compute(id, arg_1, arg_2, arg_3, 
     .           arg_4, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .           mem4lot:mem4hit, mem4loe:mem4hie, mem4lof:mem4hif)

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
      INTEGER TM_LENSTR1, vlen, tlen, slen, flen, iunit
      CHARACTER*2048 outstring, first, second, third
      CHARACTER*2048 outfile
      CHARACTER*50 tag
      CHARACTER*2048 value
      CHARACTER*100 errtxt
      INTEGER cdata_flag

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      i = arg_lo_ss(X_AXIS,ARG1)
      j = arg_lo_ss(Y_AXIS,ARG1)
      k = arg_lo_ss(Z_AXIS,ARG1)
      l = arg_lo_ss(T_AXIS,ARG1)
      m = arg_lo_ss(E_AXIS,ARG1)
      n = arg_lo_ss(F_AXIS,ARG1)
*     Get the tag (string)
      CALL ef_get_string_arg_element_6d(id, ARG1, arg_1,
     .                                  i,j,k,l,m,n,tlen,tag)

      i = arg_lo_ss(X_AXIS,ARG2)
      j = arg_lo_ss(Y_AXIS,ARG2)
      k = arg_lo_ss(Z_AXIS,ARG2)
      l = arg_lo_ss(T_AXIS,ARG2)
      m = arg_lo_ss(E_AXIS,ARG2)
      n = arg_lo_ss(F_AXIS,ARG2)
*     Get the value (string)
      CALL ef_get_string_arg_element_6d(id, ARG2, arg_2,
     .                                  i,j,k,l,m,n,vlen,value)
      i = arg_lo_ss(X_AXIS,ARG3)
      j = arg_lo_ss(Y_AXIS,ARG3)
      k = arg_lo_ss(Z_AXIS,ARG3)
      l = arg_lo_ss(T_AXIS,ARG3)
      m = arg_lo_ss(E_AXIS,ARG3)
      n = arg_lo_ss(F_AXIS,ARG3)
      cdata_flag = INT(arg_3(i,j,k,l,m,n))


      i = arg_lo_ss(X_AXIS,ARG4)
      j = arg_lo_ss(Y_AXIS,ARG4)
      k = arg_lo_ss(Z_AXIS,ARG4)
      l = arg_lo_ss(T_AXIS,ARG4)
      m = arg_lo_ss(E_AXIS,ARG4)
      n = arg_lo_ss(F_AXIS,ARG4)
*     Get the filename (string)
      CALL ef_get_string_arg_element_6d(id, ARG4, arg_4,
     .                                  i,j,k,l,m,n,flen,outfile)

*		<value><![CDATA[0]]></value>
*		<value>0</value>

1000  FORMAT('<',a, '><![CDATA[')
1010  FORMAT(a, ']]></', a, '>')
1020  FORMAT(a, '>')
1030  FORMAT('<',a, '>')
1040  FORMAT(a, '</')
1050  FORMAT(a, '>')
1060  FORMAT(a)

      IF (cdata_flag .EQ. 1) THEN
         write (first,  1000) tag(1:tlen)
         write (second, 1010) value(1:vlen)
         write (third , 1020) tag(1:tlen)
	 tlen = TM_LENSTR1(first)
	 vlen = TM_LENSTR1(second)
	 slen = TM_LENSTR1(third)
         outstring = first(1:tlen)//second(1:vlen)//third(1:slen)
	 vlen = tlen+vlen+slen
      ELSE
         write (first,  1030) tag(1:tlen)
         write (second, 1040) value(1:vlen)
         write (third , 1050) tag(1:tlen)
	 tlen = TM_LENSTR1(first)
	 vlen = TM_LENSTR1(second)
	 slen = TM_LENSTR1(third)
         outstring = first(1:tlen)//second(1:vlen)//third(1:slen)
	 vlen = tlen+vlen+slen
      ENDIF
      vlen = TM_LENSTR1(outstring)

      iunit = 19
      OPEN( UNIT        = iunit,
     .      FILE        = outfile(1:flen),
     .      FORM        = 'FORMATTED',
     .      ACCESS      = 'APPEND',
     .      ERR         = 5010 )

      WRITE (iunit, 1060) outstring(1:vlen)
      CLOSE (iunit, ERR = 5020 )

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)
      result(i,j,k,l,m,n) = 1

      RETURN

 5010 CONTINUE
      WRITE(errtxt,*) 'Error opening file ', outfile(1:flen)
      CALL EF_BAIL_OUT(id, errtxt)
 5020 CONTINUE
      WRITE(errtxt,*) 'Error closing file ', outfile(1:flen)
      CALL EF_BAIL_OUT(id, errtxt)
      RETURN

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
