
*
* index_to_grid.F
*
* Jonathan Callahan
* Feb 25th 2002
*
* This function reads in tuples (i,j,val) which correspond
* to the FORTRAN indices of the value.  Some sites use this
* method to compress data for gridded fields.
*
* The fourth argument contains the destination grid.
*
* The tuples are assumed to already be defined on an abstract
* X and real Z and T axes.  The examples I have seen so far
* are model output at a single level and multiple timesteps.
* The tuple output only corresponds to the X and Y axes and 
* is used instead of assigning bad flags or masks.
*
* Note that the FORTRAN indices I and J DO NOT correspond
* to the Ferret indices I and J.  The following interpretation
* is made:
*
* FORTRAN I = row number = FERRET J = latitude (perhaps oriented N->S)
* FORTRAN J = col number = FERRET I = longitude


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


      SUBROUTINE index_to_grid_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'puts (I,J,VAL) tuples on grid of D' )

      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'I')
      CALL ef_set_arg_desc(id, arg, 'Fortran I index (row) of VAL ')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'J')
      CALL ef_set_arg_desc(id, arg, 'Fortran J index (column) of VAL')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'VAL')
      CALL ef_set_arg_desc(id, arg, 'Value at grid cell [I,J]')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'D')
      CALL ef_set_arg_desc(id, arg, 'destination grid taken from D')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, NO)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE index_to_grid_compute(id, arg_1, arg_2, arg_3, arg_4,
     .                                 result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy,
     .     mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy,
     .     mem4loz:mem4hiz, mem4lot:mem4hit)
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
      INTEGER i3,j3,k3,l3

      INTEGER index_i,index_j

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


* First -- populate the result with bad flags.

      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                     result(i,j,k,l) = bad_flag_result

 100           CONTINUE
 200        CONTINUE
 300     CONTINUE
 400  CONTINUE
      

* Second -- Loop over the K and L axes.
*           Put the values where they belong.
*
*           Assumption: I,J,VAL are on the same grid.
*           Assumption: I is the row # and is associated with a latitude
*           Assumption: J is the col # and is associated with a longitude

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

* Note: there should not be a Y axis on argument 3
      j3 = arg_lo_ss(Y_AXIS,ARG3)

      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO 700 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         k3 = arg_lo_ss(Z_AXIS,ARG3)
         DO 600 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            DO 500 i3=arg_lo_ss(X_AXIS,ARG3), arg_hi_ss(X_AXIS,ARG3)
               index_i = i - 1 + arg_2(i3,j3,k3,l3) 
               index_j = j - 1 + arg_1(i3,j3,k3,l3)
               result(index_i,index_j,k,l) = arg_3(i3,j3,k3,l3)
 500        CONTINUE

            k3 = k3 + arg_incr(Z_AXIS,ARG3)
 600     CONTINUE
         l3 = l3 + arg_incr(T_AXIS,ARG3)
 700  CONTINUE
      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
