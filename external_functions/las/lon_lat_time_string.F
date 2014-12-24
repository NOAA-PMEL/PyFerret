*
* lon_lat_time_string.F
*
*
*  In this subroutine we provide information about
*  the function.  The user configurable information
*  consists of the following:
*
*  descr              Text description of the function
*
*  num_args           Required number of arguments
*
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_ARGS - same axis as the incoming argument
*                        NORMAL          - the result is normal to this axis
*                        ABSTRACT        - an axis which only has index values
*
*  piecemeal_ok       For memory optimization:
*                        axes where calculation may be performed piecemeal
*                        ( YES, NO )
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


      SUBROUTINE lon_lat_time_string_init(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'Generate names based on longitude,latitude,time,and a string')
      CALL ef_set_num_args(id, 5)
      CALL ef_set_axis_inheritance(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'longitude')
      CALL ef_set_arg_desc(id, arg, 'Station longitude')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'latitude')
      CALL ef_set_arg_desc(id, arg, 'Station latitude')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'time')
      CALL ef_set_arg_desc(id, arg, 'numeric times')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'time_axis')
      CALL ef_set_arg_desc(id, arg, 
     .     'Varible with a time axis to use to convert time to time strings')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'string to prepend')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'A label to start the names with')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE lon_lat_time_string_compute(id, arg_1, arg_2, arg_3, arg_4, 
     .          arg_5, result)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .           mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz,
     .           mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz,
     .           mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz,
     .           mem4lot:mem4hit)
      REAL arg_5(1,mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz,
     .           mem5lot:mem5hit)

      REAL result(1,memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information
* for the result axes.  The 'arg_' arrays will contain the indexing
* information for each variable's axes.

      INTEGER res_lo_ss(4),
     .        res_hi_ss(4),
     .        res_incr (4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS),
     .        arg_hi_ss(4,EF_MAX_ARGS),
     .        arg_incr (4,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      INTEGER TM_LENSTR
      INTEGER i,  j,  k,  l
      INTEGER i1, j1, k1, l1
      INTEGER slen,len1,len2,outlen,llen

      CHARACTER*20  TM_FMT, datestring, lonstr, latstr, ilab
      CHARACTER*256 lab, outstring
      REAL ddate

 1100  FORMAT(i1)
 1200  FORMAT(i2.2)
 1300  FORMAT(i3.3)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      IF (arg_hi_ss(X_AXIS,ARG1) .GE. 1000)  
     .  CALL ef_bail_out(id,'Function for making lists size 999 or smaller')

      CALL ef_get_arg_string(id, ARG5, lab)
      slen = TM_LENSTR(lab)

      l1 = arg_lo_ss(T_AXIS,ARG1)
        DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

          j1 = arg_lo_ss(Y_AXIS,ARG1)
          DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           i1 = arg_lo_ss(X_AXIS,ARG1)
           DO 100 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            IF ( arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1) .OR.
     .           arg_2(i1,j1,k1,l1) .EQ. bad_flag(ARG2) .OR.
     .           arg_3(i1,j1,k1,l1) .EQ. bad_flag(ARG3) ) THEN

C     Get the string value of dates and put them in the result.

               outstring = '***'
	       CALL EF_PUT_STRING (outstring, 3, result(1,i,j,k,l))

            ELSE

	       lonstr = TM_FMT(arg_1(i1,j1,k1,l1), 4, 5, len1)
	       latstr = TM_FMT(arg_2(i1,j1,k1,l1), 4, 5, len2)
	          
               ddate = arg_3(i1,j1,k1,l1)
               CALL EF_GET_AXIS_DATES (id, ARG4, ddate, 1, datestring)
               outstring = lab(:slen)//'_'//lonstr(:len1)//
     .           'E_'//latstr(:len2) //'N_'//datestring(1:11)

cc Just use a sequence number...
cc
cc	       IF (arg_hi_ss(X_AXIS,ARG1) .LT. 10) THEN
cc	          WRITE (ilab, 1100) i1
cc		  llen = 1
cc	       ELSEIF (arg_hi_ss(X_AXIS,ARG1) .LT. 100) THEN
cc	          WRITE (ilab, 1200) i1
cc		  llen = 2
cc	       ELSEIF (arg_hi_ss(X_AXIS,ARG1) .LT. 1000) THEN
cc	          WRITE (ilab, 1300) i1
cc		  llen = 3
cc	       ENDIF
cc               outstring = lab(:slen)//'_'//ilab(:llen)
     
	       outlen = TM_LENSTR(outstring)
	       CALL EF_PUT_STRING (outstring, outlen, result(1,i,j,k,l))

            ENDIF

            i1 = i1 + arg_incr(X_AXIS,ARG1)
 100       CONTINUE

           j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200      CONTINUE

          k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300     CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
 400    CONTINUE


      RETURN

 999  CALL EF_BAIL_OUT(id, 'Argument 2 is not a numeric string')

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
