* 
*  compressl_by.F
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
*  Steve Hankin
*  12/27/2000
*   7/ 9/2004 *acm* add disclaimer text
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 
* 
*  Compress argument 1 along axis L by the mask given in argument 2
*  Argument 2 may be multidimensional
 
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


      SUBROUTINE compressl_by_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'Compress arg1 by (multi-D) mask in arg 2' )

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, ABSTRACT,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'dat')
      CALL ef_set_arg_desc(id, arg, 'variable to compress on L axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'mask')
      CALL ef_set_arg_desc(id, arg,
     .          'valid/bad points guiding compression (multi-D)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END






*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE compressl_by_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_hi_l, lmsk
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      my_hi_l = arg_hi_ss(T_AXIS,1) - arg_lo_ss(T_AXIS,1) + 1

      lmsk    = arg_hi_ss(T_AXIS,2) - arg_lo_ss(T_AXIS,2) + 1
      IF ( lmsk .LT. my_hi_l ) my_hi_l = lmsk

      CALL ef_set_axis_limits(id, T_AXIS, 1, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* 
*  In this subroutine we compute the result
* 
      SUBROUTINE compressl_by_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER nout, llo

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      llo = res_lo_ss(T_AXIS)

* pre-fill the entire output result with missing value flags
*  (since each L-sequence in mask may have different number of valid pts)
* pre-fill the entire output result with missing value flags
*  (since each I-sequence in mask may have different number of valid pts)
      result = bad_flag_result

* fill in the points as specified by the mask
      n1 = arg_lo_ss(F_AXIS,ARG1)
      n2 = arg_lo_ss(F_AXIS,ARG2)
      DO 600 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         m2 = arg_lo_ss(E_AXIS,ARG2)
         DO 500 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            DO 400 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               j1 = arg_lo_ss(Y_AXIS,ARG1)
               j2 = arg_lo_ss(Y_AXIS,ARG2)
               DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

                  i1 = arg_lo_ss(X_AXIS,ARG1)
                  i2 = arg_lo_ss(X_AXIS,ARG2)
                  DO 200 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                     nout = 0
                     l1 = arg_lo_ss(T_AXIS,ARG1)
                     l2 = arg_lo_ss(T_AXIS,ARG2)
                     DO 100 l=llo, res_hi_ss(T_AXIS)

                        IF ( arg_2(i2,j2,k2,l2,m2,n2)
     .                                   .NE. bad_flag(2) ) THEN
                           IF ( arg_1(i1,j1,k1,l1,m1,n1) 
     .                                   .EQ. bad_flag(1) ) THEN
                              result(i,j,k,llo+nout,m,n) = 
     .                                   bad_flag_result
                           ELSE
                              result(i,j,k,llo+nout,m,n) = 
     .                                   arg_1(i1,j1,k1,l1,m1,n1)
                           ENDIF
                           nout = nout + 1
                        ENDIF

                        l1 = l1 + arg_incr(T_AXIS,ARG1)
                        l2 = l2 + arg_incr(T_AXIS,ARG2)
 100                 CONTINUE

                     i1 = i1 + arg_incr(X_AXIS,ARG1)
                     i2 = i2 + arg_incr(X_AXIS,ARG2)
 200              CONTINUE

                  j1 = j1 + arg_incr(Y_AXIS,ARG1)
                  j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
 400        CONTINUE

            m1 = m1 + arg_incr(E_AXIS,ARG1)
            m2 = m2 + arg_incr(E_AXIS,ARG2)
 500     CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
         n2 = n2 + arg_incr(F_AXIS,ARG2)
 600  CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

