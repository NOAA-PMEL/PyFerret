*
* samplet_date.F
*
*
* This function samples data along the L axis at a set of times: 
* yr,mo,da,hr,min,sec

* 5/20/99  Ansley Manke 
*         Times may be oriented along any axis; 
*         Result is abstract on the sampled axis.
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


      SUBROUTINE samplet_date_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to a set of times')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 7)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, ABSTRACT)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 2)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample at set of times')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YR')
      CALL ef_set_arg_desc(id, arg, 'Year yyyy')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'MO')
      CALL ef_set_arg_desc(id, arg, 'Month mm')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'DAY')
      CALL ef_set_arg_desc(id, arg, 'Day of month dd')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'HR')
      CALL ef_set_arg_desc(id, arg, 'Hour hh')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'MIN')
      CALL ef_set_arg_desc(id, arg, 'Minute mm')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'SEC')
      CALL ef_set_arg_desc(id, arg, 'Second ss')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* loss               lo subscript for an axis
*
* hiss               hi subscript for an axis
*

      SUBROUTINE samplet_date_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l
      INTEGER nx, ny, nz, nt

*     Use utility functions to get context information about the 
*     2nd argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1

*  Use the subscripts for ARG2 for the abstract axis limits. 

      my_lo_l = 1
      my_hi_l = nt
      IF (arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) .EQ. 0) 
     .    my_hi_l = MAX(nx,ny,nz,nt)

      CALL ef_set_axis_limits(id, T_AXIS, my_lo_l, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE samplet_date_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id, array #, X len, Y len, Z len, T len)
*
      INTEGER mtdat, mt2, mt5

      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

*  Use utility functions to get context information about the 1st
*  argument, to set the work arrays for the input argument time axis..

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mtdat = arg_hi_ss(T_AXIS, ARG1) - arg_lo_ss(T_AXIS, ARG1) + 1

      mt2 = mtdat* 2	! real*8  takes twice what a real*4 does
      mt5 = mtdat* 5	! character*20: character*1 takes 1 byte

*  taxdat (real*8)
      CALL ef_set_work_array_lens (id, 1, mt2, 1, 1, 1)

*  datebuf (character*20)
      CALL ef_set_work_array_lens (id, 2, mt5, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE samplet_date_compute(id, arg_1, arg_2, arg_3,
     .      arg_4, arg_5, arg_6, arg_7, result, taxdat, 
     .      datebuf)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .      mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .     mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .     mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .     mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz, 
     .     mem7lot:mem7hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

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

      INTEGER id
      INTEGER i, j, k, l
      INTEGER arg
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER i3, j3, k3, l3
      INTEGER i4, j4, k4, l4
      INTEGER i5, j5, k5, l5
      INTEGER i6, j6, k6, l6
      INTEGER i7, j7, k7, l7
      INTEGER numtimes
      INTEGER lyr, lmo, lda, lhr, lmn, lsc
      REAL ayr, amo, ada, ahr, amn, asc

      CHARACTER ax_name(4,7)*64, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

      INTEGER nlen(3)
      CHARACTER*8 lefint, aindex(3)
      CHARACTER*100 errtxt

*  set work arrays

      REAL*8 taxdat(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      CHARACTER*20 datebuf(wrk1lox:wrk1hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      CHARACTER*3 months(12), cmon
      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

 120  FORMAT (i2, 1x, a3, 1x, i4, 3(1x,i2))

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      iarg = 1
      CALL ef_get_coordinates (id, ARG1, T_AXIS, 
     .       arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdat)

      numtimes = arg_hi_ss(T_AXIS, ARG1) - arg_lo_ss(T_AXIS, ARG1) + 1
      CALL ef_get_axis_dates (id, ARG1, taxdat, numtimes, datebuf)

      DO 90 arg = 1, 7
        CALL ef_get_axis_info (id, arg, ax_name(1,arg), ax_units, 
     .     backward, modulo, regular)
  90  CONTINUE

      i1 = arg_lo_ss(X_AXIS, ARG1)
      i2 = arg_lo_ss(X_AXIS, ARG2)
      i3 = arg_lo_ss(X_AXIS, ARG3)
      i4 = arg_lo_ss(X_AXIS, ARG4)
      i5 = arg_lo_ss(X_AXIS, ARG5)
      i6 = arg_lo_ss(X_AXIS, ARG6)
      i7 = arg_lo_ss(X_AXIS, ARG7)
      DO 400 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         j3 = arg_lo_ss(Y_AXIS,ARG3)
         j4 = arg_lo_ss(Y_AXIS,ARG4)
         j5 = arg_lo_ss(Y_AXIS,ARG5)
         j6 = arg_lo_ss(Y_AXIS,ARG6)
         j7 = arg_lo_ss(Y_AXIS,ARG7)
         DO 300 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            k3 = arg_lo_ss(Z_AXIS,ARG3)
            k4 = arg_lo_ss(Z_AXIS,ARG4)
            k5 = arg_lo_ss(Z_AXIS,ARG5)
            k6 = arg_lo_ss(Z_AXIS,ARG6)
            k7 = arg_lo_ss(Z_AXIS,ARG7)
            DO 200 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l2 = arg_lo_ss(T_AXIS,ARG2)
               l3 = arg_lo_ss(T_AXIS,ARG3)
               l4 = arg_lo_ss(T_AXIS,ARG4)
               l5 = arg_lo_ss(T_AXIS,ARG5)
               l6 = arg_lo_ss(T_AXIS,ARG6)
               l7 = arg_lo_ss(T_AXIS,ARG7)
               DO 100 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)


*  Get the time to sample, whichever axis of arg_2 thru arg_7 it is on.

                  ayr = arg_2(i2,j2,k2,l2)
                  IF (ax_name(T_AXIS,2) .EQ. 'normal') 
     .                  CALL pickindex(arg_2, l, ayr)
                  lyr = ayr

                  amo = arg_3(i3,j3,k3,l3)
                  IF (ax_name(T_AXIS,3) .EQ. 'normal') 
     .                  CALL pickindex(arg_3, l, amo)
                  lmo = amo

                  ada = arg_4(i4,j4,k4,l4)
                  IF (ax_name(T_AXIS,4) .EQ. 'normal') 
     .                  CALL pickindex(arg_4, l, ada)
                  lda = ada

                  ahr = arg_5(i5,j5,k5,l5)
                  IF (ax_name(T_AXIS,5) .EQ. 'normal') 
     .                  CALL pickindex(arg_5, l, ahr)
                  lhr = ahr

                  amn = arg_6(i6,j6,k6,l6)
                  IF (ax_name(T_AXIS,6) .EQ. 'normal') 
     .                  CALL pickindex(arg_6, l, amn)
                  lmn = amn

                  asc = arg_7(i7,j7,k7,l7)
                  IF (ax_name(T_AXIS,7) .EQ. 'normal') 
     .                  CALL pickindex(arg_7, l, asc)
                  lsc = asc


*  Check that we do have an valid time.

                  IF (ayr .EQ. bad_flag(ARG2) .OR.
     .                amo .EQ. bad_flag(arg3) .OR. 
     .                ada .EQ. bad_flag(arg4) .OR. 
     .                ahr .EQ. bad_flag(arg5) .OR. 
     .                amn .EQ. bad_flag(arg6) .OR. 
     .                asc .EQ. bad_flag(arg7) ) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE

*  Find the requested time in the 1st argument, set the result.

                     result(i,j,k,l) = bad_flag_result
                     DO 600 l1 = arg_lo_ss(T_AXIS,ARG1), 
     .                           arg_hi_ss(T_AXIS,ARG1)


                        IF (arg_1(i1,j1,k1,l1) .NE. bad_flag(ARG1))THEN

                           READ (datebuf(l1,1,1,1), 120, err=900) iday, 
     .                           cmon, iyear, ihour, imin, isec
                           IF (iyear .EQ. lyr) then
                              DO 500 m = 1, 12
                                 IF (cmon .eq. months(m)) imon = m
 500                          CONTINUE
                              IF (imon  .EQ. lmo  .AND.
     .                            iday  .EQ. lda  .AND.
     .                            ihour .EQ. lhr  .AND.
     .                            imin  .EQ. lmn  .AND.
     .                            isec  .EQ. lsc) THEN
                                    result(i,j,k,l) = arg_1(i1,j1,k1,l1)
                              ENDIF 

                           ENDIF
 
                        ENDIF
 600                 CONTINUE
                  ENDIF

                  l2 = l2 + arg_incr(T_AXIS,ARG2)
                  l2 = l2 + arg_incr(T_AXIS,ARG2)
                  l3 = l3 + arg_incr(T_AXIS,ARG3)
                  l4 = l4 + arg_incr(T_AXIS,ARG4)
                  l5 = l5 + arg_incr(T_AXIS,ARG5)
                  l6 = l6 + arg_incr(T_AXIS,ARG6)
                  l7 = l7 + arg_incr(T_AXIS,ARG7)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
               k3 = k3 + arg_incr(Z_AXIS,ARG3)
               k4 = k4 + arg_incr(Z_AXIS,ARG4)
               k5 = k5 + arg_incr(Z_AXIS,ARG5)
               k6 = k6 + arg_incr(Z_AXIS,ARG6)
               k7 = k7 + arg_incr(Z_AXIS,ARG7)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
            j4 = j4 + arg_incr(Y_AXIS,ARG4)
            j5 = j5 + arg_incr(Y_AXIS,ARG5)
            j6 = j6 + arg_incr(Y_AXIS,ARG6)
            j7 = j7 + arg_incr(Y_AXIS,ARG7)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         i3 = i3 + arg_incr(X_AXIS,ARG3)
         i4 = i4 + arg_incr(X_AXIS,ARG4)
         i5 = i5 + arg_incr(X_AXIS,ARG5)
         i6 = i6 + arg_incr(X_AXIS,ARG6)
         i7 = i7 + arg_incr(X_AXIS,ARG7)
 400  CONTINUE

      RETURN 

900   CONTINUE

      WRITE (errtxt,*) 'error reading date buffer ', datebuf(l1,1,1,1)

      CALL EF_BAIL_OUT(id, errtxt)

      RETURN
999   CONTINUE

      aindex(1) = LEFINT(lorder,nlen(1))
      aindex(2) = LEFINT(arg_lo_ss(T_AXIS,ARG1),nlen(2))
      aindex(3) = LEFINT(arg_hi_ss(T_AXIS,ARG1),nlen(3))

      WRITE (errtxt,*) 'Sampling index ', aindex(1)(1:nlen(1)), 
     . ' in ARG2 is outside the range of L indices for ARG1 (', 
     .  aindex(2)(1:nlen(2)), ':', aindex(3)(1:nlen(3)), ')'

      CALL EF_BAIL_OUT(id, errtxt)

      RETURN
      END
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************
      SUBROUTINE pickindex(arr, lindex, aorder)

*  Get the index to sample, whichever axis of arg_1 it is on.  Collapse
*  the array to one dimension and return the l'th point.

      real arr(*), aorder
      integer lindex

      aorder = arr(lindex)

      return
      end
