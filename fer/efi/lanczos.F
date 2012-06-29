!
! lanczos.F90
!
! Bill Ivor Gustafson, ivor@ucdavis.edu
! 29-oct-1999
!
! This function bandpass filters the input data in time 
! using a Lanczos filter.  For details see:
!   Duchon, C. E., 1979: Lanczos filtering in one and two dimensions.
!        J. of App. Met., 18, 1016-1022.
!
! lanczos function from Brett McDaniel was in 
! Fortran 90.  Translated to f77 by Ansley Manke, PMEL  6/25/04
!   -change include files
!   -allocate weight array with a lanczos_init function
!   -change cycle and exit statements in loop 100 of 
!    compute function to GOTO's
!
! In this subroutine we provide information about
! the function.  The user configurable information 
! consists of the following:
!
! descr              Text description of the function
!
! num_args           Required number of arguments
!
! axis_inheritance   Type of axis for the result
!                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
!                       CUSTOM          - user defined axis
!                       IMPLIED_BY_ARGS - same axis as the incoming argument
!                       NORMAL          - the result is normal to this axis
!                       ABSTRACT        - an axis which only has index values
!
! piecemeal_ok       For memory optimization:
!                       axes where calculation may be performed piecemeal
!                       ( YES, NO )
! 
!
! For each argument we provide the following information:
!
! name               Text name for an argument
!
! unit               Text units for an argument
!
! desc               Text description of an argument
!
! axis_influence     Are this argument's axes the same as the result grid?
!                       ( YES, NO )
!
! axis_extend       How much does Ferret need to extend arg limits relative to result 
!


      SUBROUTINE lanczos_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

! **********************************************************************
!                                            USER CONFIGURABLE PORTION |
!                                                                      |
!                                                                      V

      CALL ef_set_desc(id, 
     .      'Bandpass filters data in time using a Lanczos filter' )

      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 
     .      'Data to be filtered in time (may also vary in x,y,z,e,f)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'F1')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,'Low frequency cutoff')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F2')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,'High frequency cutoff')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'N')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,'Number of weights (must be odd)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

!                                                                      ^
!                                                                      |
!                                            USER CONFIGURABLE PORTION |
! **********************************************************************

      RETURN 
      END  ! SUBROUTINE lanczos_init

*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE lanczos_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
*
* Set the work array X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER array_num, arg, nn
      REAL val

      arg = 4
      CALL ef_get_one_val(id, arg, val)  

      nn = val
      array_num = 1

! dimensions for weight array: nn = arg4

      CALL ef_set_work_array_dims_6d(id, array_num,
     .                                0, 0, 0, 0, 0, 0, 
     .                               nn, 0, 0, 0, 0, 0)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END ! SUBROUTINE lanczos_work_size



      SUBROUTINE lanczos_compute(id, arg_1, arg_2, arg_3, arg_4, 
     .                           result, weight)

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

      REAL weight(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .            wrk1loz:wrk1hiz, wrk1lot:wrk1hit,
     .            wrk1loe:wrk1hie, wrk1lof:wrk1hif)

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

! **********************************************************************
!                                            USER CONFIGURABLE PORTION |
!                                                                      |
!                                                                      V
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER mm, nn
      REAL aa, bb, da, f1, f2, pi, pik, sigma

      pi  = acos(-1.)

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

! Get the counter values to cycle through, nw=2*nn+1...
      i = arg_lo_ss(X_AXIS,ARG4)
      j = arg_lo_ss(Y_AXIS,ARG4)
      k = arg_lo_ss(Z_AXIS,ARG4)
      l = arg_lo_ss(T_AXIS,ARG4)
      m = arg_lo_ss(E_AXIS,ARG4)
      n = arg_lo_ss(F_AXIS,ARG4)
      if( mod(int(arg_4(i,j,k,l,m,n)),2) .EQ. 0 ) then
         CALL EF_BAIL_OUT(id,'The number of weights must be odd.')
      else
         nn = ( arg_4(i,j,k,l,m,n)-1 )/2
      endif

! Calculate the weights...

      i  = arg_lo_ss(X_AXIS,ARG2)
      j  = arg_lo_ss(Y_AXIS,ARG2)
      k  = arg_lo_ss(Z_AXIS,ARG2)
      l  = arg_lo_ss(T_AXIS,ARG2)
      m  = arg_lo_ss(E_AXIS,ARG2)
      n  = arg_lo_ss(F_AXIS,ARG2)
      f1 = arg_2(i,j,k,l,m,n)

      i  = arg_lo_ss(X_AXIS,ARG3)
      j  = arg_lo_ss(Y_AXIS,ARG3)
      k  = arg_lo_ss(Z_AXIS,ARG3)
      l  = arg_lo_ss(T_AXIS,ARG3)
      m  = arg_lo_ss(E_AXIS,ARG3)
      n  = arg_lo_ss(F_AXIS,ARG3)
      f2 = arg_3(i,j,k,l,m,n)

      weight(0,0,0,0,0,0)  = 2.*(f2-f1)
      do k = 1,nn
         pik   = pi*k
         aa    = pik/nn
         sigma = sin(aa)/aa
         aa    = 2.*pik*f1
         bb    = 2.*pik*f2
         weight(k,0,0,0,0,0) = ( sin(bb)/pik - sin(aa)/pik )*sigma
      end do

! For testing, calculate the response function...
c      da = .0005
c      aa = 0.0
c      do while( aa .LT. .5 )
c         bb = 0.
c         do i = 1,nn
c            bb = bb + weight(i,0,0,0)*cos(2.*pi*aa*float(i))
c         end do
c         bb = weight(0,0,0,0) + 2.*bb
c         write(11,*) aa,bb
c         aa = aa+da
c      end do

      if( res_hi_ss(T_AXIS) - res_lo_ss(T_AXIS) .LT. 2*nn ) then
         CALL EF_BAIL_OUT(id, 
     .     'Too many weights exist for the number of times present.')
      endif

! Apply the weights to the data...
      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 500 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 300 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 200 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         i1 = arg_lo_ss(X_AXIS,ARG1)
         DO 100 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            do l=res_lo_ss(T_AXIS),res_lo_ss(T_AXIS)+nn-1
               result(i,j,k,l,m,n) = bad_flag_result
            end do
            do l=res_hi_ss(T_AXIS)-nn+1,res_hi_ss(T_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
            end do

* Assumes result and arg_1 has same T axis dimension limits
            DO 400 l=res_lo_ss(T_AXIS)+nn, res_hi_ss(T_AXIS)-nn

               IF ( arg_1(i1,j1,k1,l,m1,n1) .EQ. bad_flag(1) ) then
                  result(i,j,k,l,m,n) = bad_flag_result
                  GOTO 400
               ENDIF
               result(i,j,k,l,m,n) = weight(0,0,0,0,0,0) *
     .                               arg_1(i1,j1,k1,l,m1,n1)

               do mm = 1,nn
                  IF (arg_1(i1,j1,k1,l-mm,m1,n1) .EQ. bad_flag(1) .OR.
     .                arg_1(i1,j1,k1,l+mm,m1,n1) .EQ. bad_flag(1)) THEN
                     result(i,j,k,l,m,n) = bad_flag_result
                     GOTO 400
                  ENDIF
                  result(i,j,k,l,m,n) = result(i,j,k,l,m,n) + 
     .                                  weight(mm,0,0,0,0,0) * 
     .                                  ( arg_1(i1,j1,k1,l-mm,m1,n1) +
     .                                    arg_1(i1,j1,k1,l+mm,m1,n1) )
               end do

 400        CONTINUE

            i1 = i1 + arg_incr(X_AXIS,ARG1)
 100     CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200  CONTINUE

      k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300  CONTINUE

      m1 = m1 + arg_incr(E_AXIS,ARG1)
 500  CONTINUE

      n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

!                                                                      ^
!                                                                      |
!                                            USER CONFIGURABLE PORTION |
! **********************************************************************

      RETURN 
      END ! SUBROUTINE lanczos_compute

