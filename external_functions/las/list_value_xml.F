*
* list_value_xml.F
*
* Ansley Manke
* May 24, 2011
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

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CHARACTER*120 fcn_descr

      WRITE (fcn_descr,100) 
      CALL ef_set_desc(id, fcn_descr)
  100 FORMAT('write a string in xml format with specified ',
     .        'tag and optional CDATA escape sequence')

      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance(id, ABSTRACT, NORMAL, 
     .			NORMAL, NORMAL)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'tag')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'tag to open and close with')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'value')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'single string value to write')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'cdata_flag')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 
     .'enter 1 to request CDATA escape, 0 otherwise')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'outputfile')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'the filename to write to')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
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

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo, my_hi, nx, ny

      INTEGER arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the argument.
*

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      CALL ef_set_axis_limits(id, Y_AXIS, 1, 3)

      my_lo = 1
      my_hi = 1

      arg = 1
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

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(2,mem1lox:mem1hix, mem1loy:mem1hiy,
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(2,mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .     mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(2,mem4lox:mem4hix, mem4loy:mem4hiy,
     .           mem4loz:mem4hiz, mem4lot:mem4hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)


* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1
      INTEGER i3, j3, k3, l3
      INTEGER TM_LENSTR1, vlen, tlen, slen
      CHARACTER*2048 outstring, first, second, third
      CHARACTER*2048 outfile
      CHARACTER*50 tag
      CHARACTER*2048 value
      CHARACTER*100 errtxt
      INTEGER cdata_flag

*  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4), tmodulo

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_result_type(id, rtype)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      j1 = arg_lo_ss(Y_AXIS,ARG1)
      k1 = arg_lo_ss(Z_AXIS,ARG1)
      l1 = arg_lo_ss(T_AXIS,ARG1)
*     Get the tag (string)
      CALL ef_get_string_arg_element(id, ARG1, arg_1,
     .     i1,j1,k1,l1,tlen,tag)

      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)
*     Get the value (string)
      CALL ef_get_string_arg_element(id, ARG2, arg_2,
     .     i2,j2,k2,l2,vlen,value)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)

      cdata_flag = INT(arg_3(i3,j3,k3,l3))


      i4 = arg_lo_ss(X_AXIS,ARG4)
      j4 = arg_lo_ss(Y_AXIS,ARG4)
      k4 = arg_lo_ss(Z_AXIS,ARG4)
      l4 = arg_lo_ss(T_AXIS,ARG4)
*     Get the value (string)
      CALL ef_get_string_arg_element(id, ARG4, arg_4,
     .     i4,j4,k4,l4,flen,outfile)


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
      ELSE
         write (first,  1030) tag(1:tlen)
         write (second, 1040) value(1:vlen)
         write (third , 1050) tag(1:tlen)
	 tlen = TM_LENSTR1(first)
	 vlen = TM_LENSTR1(second)
	 slen = TM_LENSTR1(third)
         outstring = first(1:tlen)//second(1:vlen)//third(1:slen)
      ENDIF
      vlen = TM_LENSTR1(outstring)
      


      iunit = 19

       OPEN( UNIT        = iunit,
     .       FILE        = outfile,
     .       FORM        = 'FORMATTED',
     .       ACCESS      = 'APPEND',
     .       ERR         = 5010 )

      WRITE (iunit, 1060) outstring(1:vlen)
      CLOSE (iunit, ERR = 5020 )

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      result(i,j,k,l) = 1

      RETURN
     
5010  CONTINUE
      WRITE(errtxt,*) 'Error opening file ', fname
      CALL EF_BAIL_OUT(id, errtxt)
5020  CONTINUE
      WRITE(errtxt,*) 'Error closing file ', fname
      CALL EF_BAIL_OUT(id, errtxt)
      RETURN
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
