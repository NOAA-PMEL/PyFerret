*
* convolvei.F
*
* Ansley Manke
* Feb 1999
*
* This external function convolves the component grid, com, with the weight 
* function, wt  along the I axis (see Ferret routine CONVOLVE)
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


      SUBROUTINE convolvei_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT (
     .  'Convolve component grid with weight function along I axis')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'COM')
      CALL ef_set_arg_desc(id, arg, 'Component of grid')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'WEIGHT')
      CALL ef_set_arg_desc(id, arg,
     .       'Weight function, defined on the x axis')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
	SUBROUTINE convolvei_compute (id, arg_1, arg_2, result )

*
*
*  From FERRET subroutine CONVOLVE, for the i-axis.
*
* convolve the component grid, com, with the weight function, wt
* along axis idim
* note: the component context may not be of adequate size for the full
*	calculation.  Missing data flags will be inserted where computation is
*	impossible
* also: when bad data points are encountered in the component data all
*	result data depending on it are flagged as bad, too


      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
     .     
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .      memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      REAL comp, sum
      INTEGER hlen, ii
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)
      
* CONVOLVE ALONG X AXIS (arg_1) using weights (arg_2)

* Half the weighting function; weights from -hlen to hlen.


      xlen = (arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1) 
      ylen = (arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1) 
      zlen = (arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1) 
      tlen = (arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1) 

      wlen = max(xlen, ylen, zlen, tlen)
      IF  (MOD(wlen,2) .EQ. 0) wlen = wlen + 1
      hlen = wlen/ 2

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            DO 300 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
       
               i1 = arg_lo_ss(X_AXIS,ARG1)
               DO 200 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 

                  sum = 0.0
                  i2 = arg_lo_ss(X_AXIS,ARG2)
                  j2 = arg_lo_ss(Y_AXIS,ARG2)
                  k2 = arg_lo_ss(Z_AXIS,ARG2)
                  l2 = arg_lo_ss(T_AXIS,ARG2)
                  DO 100 ii = -hlen, hlen

                     IF (i1+ii .LT. arg_lo_ss(X_AXIS,ARG1)  .OR.
     .                   i1+ii .GT. arg_hi_ss(X_AXIS,ARG1) ) THEN
                        result(i,j,k,l) = bad_flag_result
                        GOTO 190

                     ELSE
                        comp = arg_1(i1+ii,j1,k1,l1)

                        IF ( comp .EQ. bad_flag(ARG1)) THEN
                           result(i,j,k,l) = bad_flag_result
                           GOTO 190
                        ELSE
                           sum = sum + comp*arg_2(i2,j2,k2,l2)
                        ENDIF
                     ENDIF

                     i2 = i2 + arg_incr(X_AXIS,ARG2)
                     j2 = j2 + arg_incr(Y_AXIS,ARG2)
                     k2 = k2 + arg_incr(Z_AXIS,ARG2)
                     l2 = l2 + arg_incr(T_AXIS,ARG2)

 100                 CONTINUE
                     result(i,j,k,l) = sum

 190                 i1 = i1 + arg_incr(X_AXIS,ARG1)
 200              CONTINUE

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 300           CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
 400     CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
 500  CONTINUE

      RETURN
      END	
