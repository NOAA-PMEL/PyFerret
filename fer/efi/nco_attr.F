*
* nco_attr.F
*
* Ansley Manke
* March 1, 2005
*
* This function spawns a netCDF edit call, ncatted to edit 
* an attribute (e.g. new attribute, edit or delete atttibute,
* or add lines to the global history attribute)
*
* 
*  Notes: 
* 1) our ncatted does not have the i data type.  It returns the messages
* 
* > ncatted -O -h -a missing_value,A,o,i,3 x.nc 
* ncatted: ERROR `i' is not a supported netCDF data type
* ncatted: HINT: Valid data types are `c' = char, `f' = float, `d' = double,`s' = short, `l' = long, `b' = byte

* 2) errors from ncatted do not stop the function from returning normally.

      SUBROUTINE nco_attr_init(id)

* Define arguments and result

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 
     .  'Call ncatted to edit attributes in a netCDF file' )

      CALL ef_set_num_args(id, 6)
      CALL ef_set_axis_inheritance(id, ABSTRACT, NORMAL, NORMAL, 
     .                             NORMAL)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'FileName')
      CALL ef_set_arg_desc(id, arg, 'netCDF file name')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'VariableName')
      CALL ef_set_arg_desc(id, arg, 'Variable name (or global)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'AttributeName')
      CALL ef_set_arg_desc(id, arg, 'Attribute to change')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'AttType')
      CALL ef_set_arg_desc(id, arg, 'Attribute type')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'Mode')
      CALL ef_set_arg_desc(id, arg, 'o=edit, a=append, d=delete')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'AttributeValue')
      CALL ef_set_arg_desc(id, arg, 'new value of attribute')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      RETURN 
      END

* Define abstract output axis: 1 value

      SUBROUTINE nco_attr_result_limits(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER ivalue

      ivalue = 1
      CALL ef_set_axis_limits(id, X_AXIS, ivalue, ivalue)

      RETURN 
      END

*
* Compute the result
*
      SUBROUTINE nco_attr_compute(id, arg_1, arg_2, arg_3, arg_4, 
     .           arg_5, arg_6, result)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

	  INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .           mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .           mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .           mem6lot:mem6hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)
 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

      INTEGER STRINGLEN, arg, blen, slen

      REAL attval, val
      REAL*8 dattval
      INTEGER*2 isattval 
      INTEGER iattval
      CHARACTER*1 battval
      CHARACTER*1 mode, atttype
      CHARACTER*3 atttype_in
      CHARACTER*1024 filename, attributevalue
      CHARACTER*254 variablename, attributename, errmsg
      CHARACTER*2048 buff

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      arg = 1
      CALL ef_get_arg_string(id, arg, filename)
      arg = 2
      CALL ef_get_arg_string(id, arg, variablename)
      arg = 3
      CALL ef_get_arg_string(id, arg, attributename)
      arg = 4
      CALL ef_get_arg_string(id, arg, atttype_in)
      arg = 5
      CALL ef_get_arg_string(id, arg, mode)
      arg = 6
      CALL ef_get_arg_string(id, arg, attributevalue)

      IF (STRINGLEN(atttype_in) .GT. 1) THEN
         CALL EF_BAIL_OUT(id, 
     .       'Attribute type must be f, d, i, l, s, c, or b')
      ELSE
         atttype = atttype_in(1:1)
      ENDIF
      
      IF (atttype .EQ. 'i' .OR. atttype .EQ. 'l' .OR.
     .         atttype .EQ. 'I' .OR. atttype .EQ. 'L') atttype = 'l'

* Check for valid values of mode

      IF (mode .EQ. 'o' .OR. mode .EQ. 'O') THEN
         mode = 'o'
      ELSE IF (mode .EQ. 'a' .OR. mode .EQ. 'A') THEN
         mode = 'a'
      ELSE IF (mode .EQ. 'd' .OR. mode .EQ. 'D') THEN
         mode = 'd'
      ELSE
         CALL EF_BAIL_OUT(id, 'Mode must be o, a, or d')
      ENDIF

* Build the string for the ncatted command
* ncatted -O -h att_nm,var_nm,mode,att_type,att_val filename
* Mode will always be o for overwrite

      buff = 'ncatted -O -h -a '
      blen = 17

      slen = STRINGLEN(attributename)
      buff = buff(1:blen)//attributename(1:slen)//','
      blen = blen + slen + 1

      slen = STRINGLEN(variablename)
      buff = buff(1:blen)//variablename(1:slen)//','//mode(1:1)//','
      blen = blen + slen + 3

      slen = STRINGLEN(atttype)
      buff = buff(1:blen)//atttype(1:slen)//','
      blen = blen + slen + 1

! ncatted -O -h -a cartesian_axis,LAT,o,c,"Z" x.nc

* Read the attribute value based on its type.
* See documentation at the end of this file for attribute types.


      IF (atttype .EQ. 'f' .OR. atttype .EQ. 'F') THEN
         READ (attributevalue,*,err=501) attval
         slen = STRINGLEN(attributevalue)
         buff = buff(1:blen)//attributevalue(1:slen)
         blen = blen + slen

      ELSE IF (atttype .EQ. 'd' .OR. atttype .EQ. 'D') THEN
         READ (attributevalue,*,err=502) dattval
         slen = STRINGLEN(attributevalue)
         buff = buff(1:blen)//attributevalue(1:slen)
         blen = blen + slen

      ELSE IF (atttype .EQ. 'i' .OR. atttype .EQ. 'l' .OR.
     .         atttype .EQ. 'I' .OR. atttype .EQ. 'L') THEN
         READ (attributevalue,*,err=503) iattval
         READ (attributevalue,*,err=503) attval
         val = iattval
         IF (val .NE. attval) GOTO 503
         slen = STRINGLEN(attributevalue)
         buff = buff(1:blen)//attributevalue(1:slen)
         blen = blen + slen

      ELSE IF (atttype .EQ. 's' .OR. atttype .EQ. 'S') THEN
         READ (attributevalue,*,err=504) isattval
         WRITE (attributevalue,*) isattval
         slen = STRINGLEN(attributevalue)
         buff = buff(1:blen)//attributevalue(1:slen)
         blen = blen + slen

      ELSE IF (atttype .EQ. 'c' .OR. atttype .EQ. 'C') THEN
         slen = STRINGLEN(attributevalue)
         buff = buff(1:blen)//'"'//attributevalue(1:slen)//'"'
         blen = blen + slen + 3

      ELSE IF (atttype .EQ. 'b' .OR. atttype .EQ. 'B') THEN
         READ (attributevalue,*,err=505) battval
         slen = STRINGLEN(attributevalue)
         buff = buff(1:blen)//attributevalue(1:slen)
         blen = blen + slen
      ELSE
         CALL EF_BAIL_OUT(id, 
     .       'Attribute type must be f, d, i, s, c, or b')
      ENDIF

* Append the file name

      slen = STRINGLEN(filename)
      buff = buff(1:blen)//' '//filename(1:slen)
      blen = blen + slen + 1

* Spawn the command 

      result(res_lo_ss(X_AXIS), res_lo_ss(Y_AXIS), res_lo_ss(Z_AXIS), 
     .         res_lo_ss(T_AXIS)) = bad_flag_result

      CALL system( buff(1:blen) )

      result(res_lo_ss(X_AXIS), res_lo_ss(Y_AXIS), res_lo_ss(Z_AXIS), 
     .         res_lo_ss(T_AXIS)) = 1

      RETURN 

501   slen = STRINGLEN(attributevalue)
      errmsg = 'Error reading attribute value as floating-point '
     .         //attributevalue(1:slen)
      CALL EF_BAIL_OUT(id, errmsg)

502   slen = STRINGLEN(attributevalue)
      errmsg = 'Error reading attribute value as double-precision '
     .         //attributevalue(1:slen)
      CALL EF_BAIL_OUT(id, errmsg)

503   slen = STRINGLEN(attributevalue)
      errmsg = 'Error reading attribute value as an integer '
     .         //attributevalue(1:slen)
      CALL EF_BAIL_OUT(id, errmsg)

504   slen = STRINGLEN(attributevalue)
      errmsg = 'Error reading attribute value as a short integer '
     .         //attributevalue(1:slen)
      CALL EF_BAIL_OUT(id, errmsg)

505   slen = STRINGLEN(attributevalue)
      errmsg = 'Error reading attribute value as a byte '
     .         //attributevalue(1:slen)
      CALL EF_BAIL_OUT(id, errmsg)

      END


* ncatted documentation: here are the attribute types      
* f    Float. Value(s) specified in att_val will be stored as netCDF intrinsic type NC_FLOAT.
* d    Double. Value(s) specified in att_val will be stored as netCDF intrinsic type NC_DOUBLE.
* i    Integer. Value(s) specified in att_val will be stored as netCDF intrinsic type NC_INT.
* l    Long. Value(s) specified in att_val will be stored as netCDF intrinsic type NC_LONG.
* s    Short. Value(s) specified in att_val will be stored as netCDF intrinsic type NC_SHORT.
* c    Char. Value(s) specified in att_val will be stored as netCDF intrinsic type NC_CHAR.
* b    Byte. Value(s) specified in att_val will be stored as netCDF intrinsic type NC_BYTE. 
* 
* 3.1 netCDF external data types
* 
* The external types supported by the netCDF interface are:
* 
* char    8-bit characters intended for representing text.
* byte    8-bit signed or unsigned integers (see discussion below).
* short  16-bit signed integers.
* int    32-bit signed integers.
* float  real 32-bit IEEE floating-point.
* double 64-bit IEEE floating-point. 
