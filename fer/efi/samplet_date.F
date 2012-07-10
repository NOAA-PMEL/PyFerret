*
* samplet_date.F
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* This function samples data along the L axis at a set of times: 
* yr,mo,da,hr,min,sec

* 5/20/99  Ansley Manke 
*         Times may be oriented along any axis; 
*         Result is abstract on the sampled axis.
* 3/1/00  ACM fixed bug in definition of the work array datebuf
* 7/2000  ACM inherit axes only from the data variable, not dates!
* 4/2001  *ACM* add error message

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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT 
     .  ('Returns data sampled by interpolating ',
     .   'to a set of times')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 7)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, ABSTRACT,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 3)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample at set of times')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YR')
      CALL ef_set_arg_desc(id, arg, 'Year(s) yyyy')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'MO')
      CALL ef_set_arg_desc(id, arg, 'Month(s), integer mm')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'DAY')
      CALL ef_set_arg_desc(id, arg, 'Day(s) of month dd')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'HR')
      CALL ef_set_arg_desc(id, arg, 'Hour(s) hh')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'MIN')
      CALL ef_set_arg_desc(id, arg, 'Minute(s) mm')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'SEC')
      CALL ef_set_arg_desc(id, arg, 'Second(s) ss')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER my_lo_l, my_hi_l
      INTEGER nx, ny, nz, nt, ne, nf

*     Use utility functions to get context information about the 
*     2nd argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1
      ne = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1
      nf = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1

*  Use the subscripts for ARG2 for the abstract axis limits; could be on any axis

      my_lo_l = 1
      my_hi_l = MAX(nx,ny,nz,nt,ne,nf)
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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
*
* Set the work arrays, X/Y/Z/T/E/F dimensions
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER mtdat, mt2, mt20
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

*  Use utility functions to get context information about the 1st
*  argument, to set the work arrays for the input argument time axis..

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mtdat = arg_hi_ss(T_AXIS, ARG1) - arg_lo_ss(T_AXIS, ARG1) + 1
      mt2 = mtdat* 2	! real*8  takes twice what a real*4 does
      mt20 = mtdat* 20	! character*20: character*1 takes 1 byte

*  taxdat (real*8)
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              mt2, 1, 1, 1, 1, 1)

*  datebuf (character*20)
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                             mt20, 1, 1, 1, 1, 1)

*  date_axis
      CALL ef_set_work_array_dims_6d(id, 3,
     .           arg_lo_ss(T_AXIS,ARG1), 1, 1, 1, 1, 1,
     .           arg_hi_ss(T_AXIS,ARG1), 1, 1, 1, 1, 1)

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
     .                             arg_4, arg_5, arg_6, arg_7, result,
     .                            taxdat, datebuf, date_axis)

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
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .           mem5lot:mem5hit, mem5loe:mem5hie, mem5lof:mem5hif)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .           mem6lot:mem6hit, mem6loe:mem6hie, mem6lof:mem6hif)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz, 
     .           mem7lot:mem7hit, mem7loe:mem7hie, mem7lof:mem7hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

      REAL*8 taxdat(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .              wrk1loz:wrk1hiz, wrk1lot:wrk1hit,
     .              wrk1loe:wrk1hie, wrk1lof:wrk1hif)
      CHARACTER*20 datebuf(wrk2lox:wrk2hix/20, wrk2loy:wrk2hiy,
     .                     wrk2loz:wrk2hiz, wrk2lot:wrk2hit,
     .                     wrk2loe:wrk2hie, wrk2lof:wrk2hif)
      REAL date_axis(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit,
     .               wrk3loe:wrk3hie, wrk3lof:wrk3hif)

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

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER iarg
      INTEGER i, j, k, l, m, n
      INTEGER arg
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER i4, j4, k4, l4, m4, n4
      INTEGER i5, j5, k5, l5, m5, n5
      INTEGER i6, j6, k6, l6, m6, n6
      INTEGER i7, j7, k7, l7, m7, n7
      INTEGER mnth, numtimes
      INTEGER lbot, ltop, status
      REAL ayr, amo, ada, ahr, amn, asc
      REAL byr, bmo, bda, bhr, bmn, bsc
      REAL tbot, ttop, timepoint, frac

      CHARACTER*100 errtxt

      CHARACTER*3 months(12), cmon
      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

  20  FORMAT (f2.0, 1x, a3, 1x, f4.0, 3(1x,f2.0))

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      iarg = 1
      CALL ef_get_coordinates (id, ARG1, T_AXIS, 
     .       arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdat)

      numtimes = arg_hi_ss(T_AXIS, ARG1) - arg_lo_ss(T_AXIS, ARG1) + 1
      CALL ef_get_axis_dates (id, ARG1, taxdat, numtimes, datebuf)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO 60 l = 1, numtimes
         READ (datebuf(l,1,1,1,1,1), 20, err=900) bda, cmon, byr,
     .                                            bhr, bmn, bsc
         bmo = 0
         DO 40 mnth = 1, 12
            IF ( cmon .eq. months(mnth) ) bmo = mnth
   40    CONTINUE
         CALL day_of_year(bmo, bda, byr, status, errtxt)
         IF (status .NE. 0) GO TO 1000
         bmn = bmn + bsc/ 60.
         bhr = bhr + bmn/ 60.
         bda = bda + bhr/ 24.
         date_axis(l1,1,1,1,1,1) = byr + bda / 365.2425
         l1 = l1 + arg_incr(T_AXIS,ARG1)
   60 CONTINUE

      l2 = arg_lo_ss(T_AXIS,ARG2)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      l4 = arg_lo_ss(T_AXIS,ARG4)
      l5 = arg_lo_ss(T_AXIS,ARG5)
      l6 = arg_lo_ss(T_AXIS,ARG6)
      l7 = arg_lo_ss(T_AXIS,ARG7)
      DO  400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

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

            CALL day_of_year(amo, ada, ayr, status, errtxt)
            IF (status .NE. 0) GO TO 1000

            amn = amn + asc/ 60.
            ahr = ahr + amn/ 60.
            ada = ada + ahr/ 24.
            timepoint = ayr + ada / 365.2425

*  Search the 1st argument for the points before and after the requested time,
*  interpolate to set the result.

            DO 80 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
               IF (timepoint .GE. date_axis(l1,1,1,1,1,1)) lbot = l1
   80       CONTINUE
            ltop = lbot + 1

            IF (lbot .GE. arg_lo_ss(T_AXIS,ARG1)  .AND. 
     .          ltop .LE. arg_hi_ss(T_AXIS,ARG1)) THEN

               tbot = date_axis(lbot,1,1,1,1,1)
               ttop = date_axis(ltop,1,1,1,1,1)
               frac = (timepoint - tbot) / (ttop - tbot)
            ENDIF

         ENDIF

*  Loop over all space and interpolate.
  
         n1 = arg_lo_ss(F_AXIS,ARG1)
         n2 = arg_lo_ss(F_AXIS,ARG2)
         n3 = arg_lo_ss(F_AXIS,ARG3)
         n4 = arg_lo_ss(F_AXIS,ARG4)
         n5 = arg_lo_ss(F_AXIS,ARG5)
         n6 = arg_lo_ss(F_AXIS,ARG6)
         n7 = arg_lo_ss(F_AXIS,ARG7)
         DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         m2 = arg_lo_ss(E_AXIS,ARG2)
         m3 = arg_lo_ss(E_AXIS,ARG3)
         m4 = arg_lo_ss(E_AXIS,ARG4)
         m5 = arg_lo_ss(E_AXIS,ARG5)
         m6 = arg_lo_ss(E_AXIS,ARG6)
         m7 = arg_lo_ss(E_AXIS,ARG7)
         DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         k2 = arg_lo_ss(Z_AXIS,ARG2)
         k3 = arg_lo_ss(Z_AXIS,ARG3)
         k4 = arg_lo_ss(Z_AXIS,ARG4)
         k5 = arg_lo_ss(Z_AXIS,ARG5)
         k6 = arg_lo_ss(Z_AXIS,ARG6)
         k7 = arg_lo_ss(Z_AXIS,ARG7)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            j1 = arg_lo_ss(Y_AXIS,ARG1)
            j2 = arg_lo_ss(Y_AXIS,ARG2)
            j3 = arg_lo_ss(Y_AXIS,ARG3)
            j4 = arg_lo_ss(Y_AXIS,ARG4)
            j5 = arg_lo_ss(Y_AXIS,ARG5)
            j6 = arg_lo_ss(Y_AXIS,ARG6)
            j7 = arg_lo_ss(Y_AXIS,ARG7)
            DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

               i1 = arg_lo_ss(X_AXIS, ARG1)
               i2 = arg_lo_ss(X_AXIS, ARG2)
               i3 = arg_lo_ss(X_AXIS, ARG3)
               i4 = arg_lo_ss(X_AXIS, ARG4)
               i5 = arg_lo_ss(X_AXIS, ARG5)
               i6 = arg_lo_ss(X_AXIS, ARG6)
               i7 = arg_lo_ss(X_AXIS, ARG7)
               DO  100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 

                  IF ( lbot .GE. arg_lo_ss(T_AXIS,ARG1) .AND. 
     .                 ltop .LE. arg_hi_ss(T_AXIS,ARG1) ) THEN
                     IF ( arg_1(i1,j1,k1,lbot,m1,n1) 
     .                          .NE. bad_flag(ARG1)     .AND.
     .                    arg_1(i1,j1,k1,ltop,m1,n1)
     .                          .NE. bad_flag(ARG1) )     THEN
                        result(i,j,k,l,m,n) = 
     .                        arg_1(i1,j1,k1,lbot,m1,n1) + 
     .                        frac * ( arg_1(i1,j1,k1,ltop,m1,n1) -
     .                                 arg_1(i1,j1,k1,lbot,m1,n1)  )
                     ELSE
                        result(i,j,k,l,m,n) = bad_flag_result
                     ENDIF
                  ELSE
                     result(i,j,k,l,m,n) = bad_flag_result
                  ENDIF

                  i1 = i1 + arg_incr(X_AXIS,ARG1)
                  i2 = i2 + arg_incr(X_AXIS,ARG2)
                  i3 = i3 + arg_incr(X_AXIS,ARG3)
                  i4 = i4 + arg_incr(X_AXIS,ARG4)
                  i5 = i5 + arg_incr(X_AXIS,ARG5)
                  i6 = i6 + arg_incr(X_AXIS,ARG6)
                  i7 = i7 + arg_incr(X_AXIS,ARG7)
  100          CONTINUE

               j1 = j1 + arg_incr(Y_AXIS,ARG1)
               j2 = j2 + arg_incr(Y_AXIS,ARG2)
               j3 = j3 + arg_incr(Y_AXIS,ARG3)
               j4 = j4 + arg_incr(Y_AXIS,ARG4)
               j5 = j5 + arg_incr(Y_AXIS,ARG5)
               j6 = j6 + arg_incr(Y_AXIS,ARG6)
               j7 = j7 + arg_incr(Y_AXIS,ARG7)
  200       CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
            k2 = k2 + arg_incr(Z_AXIS,ARG2)
            k3 = k3 + arg_incr(Z_AXIS,ARG3)
            k4 = k4 + arg_incr(Z_AXIS,ARG4)
            k5 = k5 + arg_incr(Z_AXIS,ARG5)
            k6 = k6 + arg_incr(Z_AXIS,ARG6)
            k7 = k7 + arg_incr(Z_AXIS,ARG7)
  300    CONTINUE

            m1 = m1 + arg_incr(E_AXIS,ARG1)
            m2 = m2 + arg_incr(E_AXIS,ARG2)
            m3 = m3 + arg_incr(E_AXIS,ARG3)
            m4 = m4 + arg_incr(E_AXIS,ARG4)
            m5 = m5 + arg_incr(E_AXIS,ARG5)
            m6 = m6 + arg_incr(E_AXIS,ARG6)
            m7 = m7 + arg_incr(E_AXIS,ARG7)
  500    CONTINUE

            n1 = n1 + arg_incr(F_AXIS,ARG1)
            n2 = n2 + arg_incr(F_AXIS,ARG2)
            n3 = n3 + arg_incr(F_AXIS,ARG3)
            n4 = n4 + arg_incr(F_AXIS,ARG4)
            n5 = n5 + arg_incr(F_AXIS,ARG5)
            n6 = n6 + arg_incr(F_AXIS,ARG6)
            n7 = n7 + arg_incr(F_AXIS,ARG7)
  600    CONTINUE

* End of loop over all space; go on to next specified time

         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         l4 = l4 + arg_incr(T_AXIS,ARG4)
         l5 = l5 + arg_incr(T_AXIS,ARG5)
         l6 = l6 + arg_incr(T_AXIS,ARG6)
         l7 = l7 + arg_incr(T_AXIS,ARG7)
  400 CONTINUE

      RETURN 

  900 WRITE (errtxt,*) 'error reading date buffer ', 
     .                 l, datebuf(l1,1,1,1,1,1)
 1000 CALL EF_BAIL_OUT(id, errtxt)
      RETURN
      END
