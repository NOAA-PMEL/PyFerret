*
* sortl.F
*
* Ansley Manke
* April 1998
*
* This function sorts data on T axis in increasing order.  
* Returns index of sorted values.  The indices of any missing
* data are listed first.
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


      SUBROUTINE sortl_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns indices of data, sorted on the L axis')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 1)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, ABSTRACT)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT')
      CALL ef_set_arg_desc(id, arg, 'variable to sort in L')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)
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
* lo_ss               lo subscript for an axis
* hi_ss               hi subscript for an axis
*

      SUBROUTINE sortl_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      INTEGER my_lo_l, my_hi_l

      INTEGER arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


*     Use utility functions to get context information about the arguments.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      my_lo_l = 1
      my_hi_l = 1

      arg = 1
      my_hi_l = arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1

      CALL ef_set_axis_limits(id, T_AXIS, my_lo_l, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE sortl_compute(id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
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
      INTEGER m, m_indx, nsrt, nbad
      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1

      REAL sort_dat(100000)
      INTEGER sort_indx(100000)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)
 
      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 600 i= res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 

         j1 = arg_lo_ss(Y_AXIS, ARG1)
         DO 500 j= res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS, ARG1)
            DO 400 k= res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               nsrt = 0
               nbad = 0

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  IF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(1)) THEN
                     nbad = nbad + 1

                  ELSE
                     nsrt = nsrt + 1
                     sort_dat(nsrt) = arg_1(i1,j1,k1,l1)
                     sort_indx(nsrt) = l1

                  END IF

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE
      
*
* Sort based on sort_dat.  sort_indx goes along for the ride,
* elements moved when elements of sort_dat moved.
*
               IF (nsrt .GT. 1) CALL shell (sort_dat, sort_indx, nsrt)

*
* Put sorted data in the array first, then bad flags.
*

               DO 200 m = 1, nsrt
                  result(i,j,k,m) = float(sort_indx(m))
 200           CONTINUE

               DO 300 m = 1, nbad
                  m_indx = m + nsrt
                  result(i,j,k,m_indx) = bad_flag_result
 300           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS, ARG1)
 400        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS, ARG1)
 500     CONTINUE

      i1 = i1 + arg_incr(X_AXIS,ARG1)
 600  CONTINUE

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END


      SUBROUTINE SHELL(SDAT, IDAT,N)


*from http://www.cs.mcgill.ca/~ratzer/progs15_6.html  23-APR-1998
* Changed data to REAL. Added IDAT, to sort along with SDAT.  
* Removed NCOUNTS, NCOMP, NSWAP.

      REAL     SDAT(*)
      INTEGER  IDAT(*), N, M
      INTEGER  I, J
      M=N
      DO WHILE (M .gt. 1)
         M=(M+2)/3
         DO I=M+1,N
            DO J=I,M+1,-M
               IF(SDAT(J-M) .ge. SDAT(J)) then
                 CALL SWAPR(SDAT,J,J-M)
                 CALL SWAPI(IDAT,J,J-M)
               endif
            END DO 
         END DO 
      END DO
      RETURN
      END !SUBROUTINE SHELL
!

      SUBROUTINE SWAPI(IDAT,K,L)
      INTEGER IDAT(*),K,L
      INTEGER M
      M=IDAT(K)
      IDAT(K)=IDAT(L)
      IDAT(L)=M
      RETURN
      END !SUROUTINE SWAPI

      SUBROUTINE SWAPR(SDAT,K,L)
      REAL SDAT(*), S
      INTEGER K,L
      S=SDAT(K)
      SDAT(K)=SDAT(L)
      SDAT(L)=S
      RETURN
      END !SUROUTINE SWAP
