*
*  ann_trend.f
*
* ansley manke
* oct 1998
*
* Remove annual variation and trend from  a 2-d field.  Uses Jimmy Larsen's 
* time series code.
*


*
* in this subroutine we provide information about
* the function.  the user configurable information 
* consists of the following:
*
* descr              text description of the function
*
* num_args           required number of arguments
*
* axis_inheritance   type of axis for the result
*                       ( custom, implied_by_args, normal, abstract )
*                       custom          - user defined axis
*                       implied_by_args - same axis as the incoming argument
*                       normal          - the result is normal to this axis
*                       abstract        - an axis which only has index values
*
* piecemeal_ok       for memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( yes, no )
* 
*
* for each argument we provide the following information:
*
* name               text name for an argument
*
* unit               text units for an argument
*
* desc               text description of an argument
*
* axis_influence     are this argument's axes the same as the result grid?
*                       ( yes, no )
*
* axis_extend       how much does ferret need to extend arg limits relative to result 
*


      SUBROUTINEann_trend_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            user configurable portion |
*                                                                      |
*                                                                      v

      CALL ef_set_desc(id,
     .  'compute eofs and time functions for x-y field w/gaps' )

      CALL ef_set_num_args(id, 1)

      CALL ef_set_axis_inheritance(id, implied_by_args, 
     .      implied_by_args, implied_by_args, implied_by_args)

      CALL ef_set_piecemeal_ok(id, no, no, no, no)
 
      arg = 1
      CALL ef_set_arg_name(id, arg, 'a')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'variable in x,y,t')
      CALL ef_set_axis_influence(ID, ARG, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            user configurable portion |
************************************************************************

      RETURN 
      END



*
* in this subroutine we compute the result
*

      SUBROUTINE ann_trend_compute(id, arg_1, result)

*  arg_1  variable, function of (x,y,t)
*  result is the same variable with trend and annual variation removed

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      real bad_flag(ef_max_args), bad_flag_result
      real arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)

      real result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* after initialization, the 'res_' arrays contain indexing information 
* for the result axes.  the 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,ef_max_args), arg_hi_ss(4,ef_max_args),
     .     arg_incr(4,ef_max_args)


      INTEGER nwrk, nt
      INTEGER i, j, k, l, i1, j1, k1, l1
      REAL t_inc
      PARAMETER (nwrk=1000)
      REAL r(nwrk), gap(nwrk), fq, ssttmp(nwrk)
      REAL w(50)
      INTEGER nf, nyr
      PARAMETER (nf=5, nyr=24)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      t_inc = res_incr(t_axis)
      IF (t_inc .eq. 0) t_inc = 1

      nt = (res_hi_ss(t_axis) - res_lo_ss(t_axis) + 1)/ t_inc
      IF (nt .gt. nwrk) THEN
         PRINT *, ' ann_trend.F: set dimension for work arrays >=', nt
         GO TO 999
      ENDIF

      fq = 1.0 / (7200.0*24.*365.25/12.0)

      CALL prolate(w, nyr, 1)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS), 
     .               res_incr(Y_AXIS)

         i1 = arg_lo_ss(X_AXIS, ARG1)
         DO 400 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS), 
     .               res_incr(X_AXIS)
 
            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k = res_lo_ss(Z_AXIS)
            nt = 1

            l1 = arg_lo_ss(T_AXIS, ARG1)
            DO 100 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS), 
     .               res_incr(T_AXIS)

               ssttmp(nt) = arg_1(i1,j1,k1,l1) 
               IF (ssttmp(nt) .eq. bad_flag(ARG1)) THEN
                   ssttmp(nt) = bad_flag_result
               ELSE 
                   ssttmp(nt) = arg_1(i1,j1,k1,l1) 
               ENDIF

               nt = nt + 1

               l1 = l1 + arg_incr(T_AXIS,ARG1)
  100       CONTINUE

            nt = nt - 1
            CALL annualdai (fq, nt, ssttmp, r, gap, nf, 
     .             bad_flag_result)
            CALL trendflag2 (nt, ssttmp, bad_flag_result)
            CALL piw (ssttmp, r, w, nt, bad_flag_result)

            nt = 1
            DO 200 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
               result(i,j,k,l) = ssttmp(nt)
               nt = nt + 1
  200       CONTINUE


            i1 = i1 + arg_incr(X_AXIS,ARG1)
  400    CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
  500 CONTINUE

      return

  999 continue
      return

      end

      
*                                                                      ^
*                                                                      |
*                                            user configurable portion |
************************************************************************
