*
* samplet_date.F
*
*
* This function samples data along the L axis at a set of times: 
* yr,mo,da,hr,min,sec

* 5/20/99  Ansley Manke 
*         Times may be oriented along any axis; 
*         Result is abstract on the sampled axis.
* 3/1/00  ACM fixed bug in definition of the work array datebuf
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
   10 FORMAT 
     .  ('Returns data sampled by interpolating to a set of times')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 7)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, ABSTRACT)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 3)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample at set of times')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YR')
      CALL ef_set_arg_desc(id, arg, 'Year(s) yyyy')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'MO')
      CALL ef_set_arg_desc(id, arg, 'Month(s), integer mm')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'DAY')
      CALL ef_set_arg_desc(id, arg, 'Day(s) of month dd')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'HR')
      CALL ef_set_arg_desc(id, arg, 'Hour(s) hh')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'MIN')
      CALL ef_set_arg_desc(id, arg, 'Minute(s) mm')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'SEC')
      CALL ef_set_arg_desc(id, arg, 'Second(s) ss')
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
*
      INTEGER mtdat, mt2, mt20

      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

*  Use utility functions to get context information about the 1st
*  argument, to set the work arrays for the input argument time axis..

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mtdat = arg_hi_ss(T_AXIS, ARG1) - arg_lo_ss(T_AXIS, ARG1) + 1

      mt2 = mtdat* 2	! real*8  takes twice what a real*4 does
      mt20 = mtdat* 20	! character*20: character*1 takes 1 byte

*  taxdat (real*8)
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, mt2, 1, 1, 1)

*  datebuf (character*20)
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, mt20, 1, 1, 1)

*  date_axis
      CALL ef_set_work_array_dims (id, 3, arg_lo_ss(T_AXIS,ARG1), 
     .                   1, 1, 1, arg_hi_ss(T_AXIS, ARG1), 1, 1, 1)

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
      INTEGER iarg
      INTEGER i, j, k, l, m
      INTEGER arg
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER i3, j3, k3, l3
      INTEGER i4, j4, k4, l4
      INTEGER i5, j5, k5, l5
      INTEGER i6, j6, k6, l6
      INTEGER i7, j7, k7, l7
      INTEGER numtimes
      INTEGER lbot, ltop
      REAL ayr, amo, ada, ahr, amn, asc
      REAL byr, bmo, bda, bhr, bmn, bsc
      REAL tbot, ttop, timepoint, frac

      CHARACTER ax_name(4,7)*64, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

      CHARACTER*100 errtxt

*  set work arrays

      REAL*8 taxdat(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      CHARACTER*20 datebuf(wrk2lox:wrk2hix/20, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL date_axis(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)

      CHARACTER*3 months(12), cmon
      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

 120  FORMAT (f2.0, 1x, a3, 1x, f4.0, 3(1x,f2.0))

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      iarg = 1
      CALL ef_get_coordinates (id, ARG1, T_AXIS, 
     .       arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdat)

      numtimes = arg_hi_ss(T_AXIS, ARG1) - arg_lo_ss(T_AXIS, ARG1) + 1
      CALL ef_get_axis_dates (id, ARG1, taxdat, numtimes, datebuf)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO 200 l = 1, numtimes
         READ (datebuf(l,1,1,1), 120, err=900) bda, cmon, byr, 
     .                                         bhr, bmn, bsc

         DO 100 m = 1, 12
            IF (cmon .eq. months(m)) bmo = m
 100                  CONTINUE
         call day_of_year(bmo, bda, byr)
         bmn = bmn + bsc/ 60.
         bhr = bhr + bmn/ 60.
         bda = bda + bhr/ 24.
         date_axis(l1,1,1,1) = byr + bda/ 365.24
         l1 = l1 + arg_incr(T_AXIS,ARG1)
  200  CONTINUE

       DO 300 arg = 1, 7
          CALL ef_get_axis_info (id, arg, ax_name(1,arg), ax_units, 
     .          backward, modulo, regular)
  300  CONTINUE


      l2 = arg_lo_ss(T_AXIS,ARG2)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      l4 = arg_lo_ss(T_AXIS,ARG4)
      l5 = arg_lo_ss(T_AXIS,ARG5)
      l6 = arg_lo_ss(T_AXIS,ARG6)
      l7 = arg_lo_ss(T_AXIS,ARG7)
      DO  800 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

*  Get the time to sample, whichever axis of arg_2 thru arg_7 it is on.

         CALL pickindex(arg_2, l, ayr)
         CALL pickindex(arg_3, l, amo)
         CALL pickindex(arg_4, l, ada)
         CALL pickindex(arg_5, l, ahr)
         CALL pickindex(arg_6, l, amn)
         CALL pickindex(arg_7, l, asc)

*  Check that we do have a valid time.

         lbot = 0
         IF (ayr .NE. bad_flag(ARG2) .AND. 
     .       amo .NE. bad_flag(arg3) .AND. 
     .       ada .NE. bad_flag(arg4) .AND. 
     .       ahr .NE. bad_flag(arg5) .AND. 
     .       amn .NE. bad_flag(arg6) .AND. 
     .       asc .NE. bad_flag(arg7) ) THEN

            CALL day_of_year(amo, ada, ayr)
            amn = amn + asc/ 60.
            ahr = ahr + amn/ 60.
            ada = ada + ahr/ 24.
            timepoint = ayr + ada/ 365.25

*  Search the 1st argument for the points before and after the requested time,
*  interpolate to set the result.

            DO 400 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
               IF (timepoint .GE. date_axis(l1,1,1,1)) lbot = l1
 400        CONTINUE
            ltop = lbot + 1

            IF (lbot .GE. arg_lo_ss(T_AXIS,ARG1)  .AND. 
     .          ltop .LE. arg_hi_ss(T_AXIS,ARG1)) THEN

               tbot = date_axis(lbot,1,1,1)
               ttop = date_axis(ltop,1,1,1)
               frac = (timepoint - tbot ) / (ttop - tbot)
            ENDIF

         ENDIF

*  Loop over all space and interpolate.
  
         i1 = arg_lo_ss(X_AXIS, ARG1)
         i2 = arg_lo_ss(X_AXIS, ARG2)
         i3 = arg_lo_ss(X_AXIS, ARG3)
         i4 = arg_lo_ss(X_AXIS, ARG4)
         i5 = arg_lo_ss(X_AXIS, ARG5)
         i6 = arg_lo_ss(X_AXIS, ARG6)
         i7 = arg_lo_ss(X_AXIS, ARG7)
         DO  700 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 

            j1 = arg_lo_ss(Y_AXIS,ARG1)
            j2 = arg_lo_ss(Y_AXIS,ARG2)
            j3 = arg_lo_ss(Y_AXIS,ARG3)
            j4 = arg_lo_ss(Y_AXIS,ARG4)
            j5 = arg_lo_ss(Y_AXIS,ARG5)
            j6 = arg_lo_ss(Y_AXIS,ARG6)
            j7 = arg_lo_ss(Y_AXIS,ARG7)
            DO 600 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

               k1 = arg_lo_ss(Z_AXIS,ARG1)
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               k3 = arg_lo_ss(Z_AXIS,ARG3)
               k4 = arg_lo_ss(Z_AXIS,ARG4)
               k5 = arg_lo_ss(Z_AXIS,ARG5)
               k6 = arg_lo_ss(Z_AXIS,ARG6)
               k7 = arg_lo_ss(Z_AXIS,ARG7)
               DO 500 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  IF ( (lbot .GE. arg_lo_ss(T_AXIS,ARG1)   .AND. 
     .                  ltop .LE. arg_hi_ss(T_AXIS,ARG1) ) .AND.
     .                 (arg_1(i1,j1,k1,lbot) .NE. bad_flag(ARG1) .AND.
     .                  arg_1(i1,j1,k1,ltop) .NE. bad_flag(ARG1)) ) THEN

                     result(i,j,k,l) = arg_1(i1,j1,k1,lbot) + frac* 
     .                   (arg_1(i1,j1,k1,ltop) - arg_1(i1,j1,k1,lbot))

                  ELSE
                     result(i,j,k,l) = bad_flag_result
                  ENDIF

                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
                  k2 = k2 + arg_incr(Z_AXIS,ARG2)
                  k3 = k3 + arg_incr(Z_AXIS,ARG3)
                  k4 = k4 + arg_incr(Z_AXIS,ARG4)
                  k5 = k5 + arg_incr(Z_AXIS,ARG5)
                  k6 = k6 + arg_incr(Z_AXIS,ARG6)
                  k7 = k7 + arg_incr(Z_AXIS,ARG7)
 500           CONTINUE

               j1 = j1 + arg_incr(Y_AXIS,ARG1)
               j2 = j2 + arg_incr(Y_AXIS,ARG2)
               j3 = j3 + arg_incr(Y_AXIS,ARG3)
               j4 = j4 + arg_incr(Y_AXIS,ARG4)
               j5 = j5 + arg_incr(Y_AXIS,ARG5)
               j6 = j6 + arg_incr(Y_AXIS,ARG6)
               j7 = j7 + arg_incr(Y_AXIS,ARG7)
 600        CONTINUE

            i1 = i1 + arg_incr(X_AXIS,ARG1)
            i2 = i2 + arg_incr(X_AXIS,ARG2)
            i3 = i3 + arg_incr(X_AXIS,ARG3)
            i4 = i4 + arg_incr(X_AXIS,ARG4)
            i5 = i5 + arg_incr(X_AXIS,ARG5)
            i6 = i6 + arg_incr(X_AXIS,ARG6)
            i7 = i7 + arg_incr(X_AXIS,ARG7)
  700    CONTINUE

         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         l4 = l4 + arg_incr(T_AXIS,ARG4)
         l5 = l5 + arg_incr(T_AXIS,ARG5)
         l6 = l6 + arg_incr(T_AXIS,ARG6)
         l7 = l7 + arg_incr(T_AXIS,ARG7)
  800 CONTINUE

      RETURN 

900   CONTINUE

      WRITE (errtxt,*) 'error reading date buffer ',l, datebuf(l1,1,1,1)

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


      subroutine day_of_year(amo, ada, ayr)

C  Find sequential day of the year corresponding to month, day, year. 
C  Return sequential day in variable ada.

      REAL daymo(12)           ! Number of days in each month
      DATA daymo/31,28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/
      REAL amo, ada, ayr
      INTEGER iyr, imo, i

      iyr = ayr
      IF (MOD(iyr, 4) .EQ. 0) THEN
        daymo(2) = 29
      ENDIF

      imo = amo
      DO 100 i = 1, imo-1
       ada = ada + daymo(i)
  100 CONTINUE

      RETURN
      END
