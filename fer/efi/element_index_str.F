*
* element_index_str_n.F
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
* Ansley Manke
* July 2005
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*
*  Return index value in LIST for each grids point in V, or a
*  missing value wherever there is no match)
*
*   3/12 *kms* Give index into unraveled arg_2 instead of just I index of arg_2;
*              increased max string length compared from 20 to 128;
*              add check for string lengths .GT. 0;
*              and add E and F dimensions
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
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


      SUBROUTINE element_index_str_n_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT (
     .'Return index value in ARG2 for each string in ARG1 ',
     .'(case sensitive matching)')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)


      arg = 1
      CALL ef_set_arg_name(id, arg, 'VAR')
      CALL ef_set_arg_desc(id, arg, 'string data to mask')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'STRINGS')
      CALL ef_set_arg_desc(id, arg, 'list of strings to match')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE element_index_str_n_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER strdf
#ifdef double_p
      PARAMETER (strdf = 1)
#else
      PARAMETER (strdf = 2)
#endif

      INTEGER id

      REAL arg_1(strdf,
     .           mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(strdf,
     .           mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

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
      INTEGER arg, ndx
      INTEGER i, j, k, l, m, n
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER slen1, slen2
      CHARACTER text1*128, text2*128


      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

C  Check for strings matching arg2
C  Assumes result and arg_1 have same array dimension limits

      DO n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      DO k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
      DO j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      DO i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         arg = 1
         CALL ef_get_string_arg_element_6d(id, arg, arg_1, 
     .                      i, j, k, l, m, n, slen1, text1)
 
         ndx = 1
         DO n2=arg_lo_ss(F_AXIS,ARG2), arg_hi_ss(F_AXIS,ARG2)
         DO m2=arg_lo_ss(E_AXIS,ARG2), arg_hi_ss(E_AXIS,ARG2)
         DO l2=arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)
         DO k2=arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)
         DO j2=arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
         DO i2=arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
            arg = 2
            CALL ef_get_string_arg_element_6d(id, arg, arg_2, 
     .                         i2, j2, k2, l2, m2, n2, slen2, text2)
            IF ( slen1 .GT. 0 .AND.
     .           slen2 .GT. 0 .AND.
     .           slen1 .EQ. slen2 ) THEN
              IF ( text1(:slen1) .EQ. text2(:slen2) ) THEN
                 result(i,j,k,l,m,n) = ndx
                 GOTO 200
              ENDIF
            ENDIF
            ndx = ndx + 1
         ENDDO
         ENDDO
         ENDDO
         ENDDO
         ENDDO
         ENDDO

C        If we got here it was not found
         result(i,j,k,l,m,n) = bad_flag_result

 200     CONTINUE

      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
 
      RETURN
      END
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************
