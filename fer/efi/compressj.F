*
* compressj.F
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
* August 1998
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
* This function compresses data on J axis, ending with missing data.
*
*  NOTE:
*  IT IS GENERALLY ADVISABLE TO INCLUDE EXPLICIT LIMITS WHEN WORKING WITH
*  FUNCTIONS THAT REPLACE AXES. FOR EXAMPLE, THE CONSIDER THE FUNCTION
*  COMPRESSL(v). THE EXPRESSION
*  	LIST/L=6:10 COMPRESSL(v)
*  IS NOT EQUIVALENT TO
*  	LIST COMPRESSL(v[L=6:10])
*  THE FORMER WILL LIST THE 6TH THROUGH 10TH COMPRESSED INDICES FROM THE ENTIRE
*  L RANGE OF VARIABLE V. THE LATTER WILL LIST ALL OF THE INDICES THAT
*  RESULT FROM COMPRESSING v[L=6:10].
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


      SUBROUTINE compressj_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data, compressed along the J axis: Missing ',
     .        'points moved to the end')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 1)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, ABSTRACT, 
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT')
      CALL ef_set_arg_desc(id, arg, 'variable to compress in J')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, YES, YES, YES, YES)

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
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE compressj_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l

      INTEGER arg
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      my_lo_l = 1
      my_hi_l = 1

      arg = 1

      my_hi_l = arg_hi_ss(Y_AXIS,arg) - arg_lo_ss(Y_AXIS,arg) + 1


      CALL ef_set_axis_limits(id, Y_AXIS, my_lo_l, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE compressj_compute(id, arg_1, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)

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


***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER ndx, nsrt, nbad
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1

*  Get subscript ranges

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      n1 = arg_lo_ss(F_AXIS, ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS, ARG1)
         DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            l1 = arg_lo_ss(T_AXIS, ARG1)
            DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               k1 = arg_lo_ss(Z_AXIS, ARG1)
               DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  i1 = arg_lo_ss(X_AXIS, ARG1)
                  DO 200 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                     nsrt = 0
                     nbad = 0

                     j1 = arg_lo_ss(Y_AXIS,ARG1)
                     DO 100 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

                        IF ( arg_1(i1,j1,k1,l1,m1,n1)
     .                               .NE. bad_flag(1) ) THEN
                           nsrt = nsrt + 1
                           result(i,nsrt,k,l,m,n) = 
     .                               arg_1(i1,j1,k1,l1,m1,n1)
                        ELSE
                           nbad = nbad + 1
                        ENDIF

                        j1 = j1 + arg_incr(Y_AXIS,ARG1)
 100                 CONTINUE
      
*
* Compress argument, putting bad data at the end.
*
                     DO 150 ndx = 1, nbad
                        nsrt = nsrt + 1
                        result(i,nsrt,k,l,m,n) = bad_flag_result
 150                 CONTINUE

                     i1 = i1 + arg_incr(X_AXIS, ARG1)
 200              CONTINUE

                  k1 = k1 + arg_incr(Z_AXIS, ARG1)
 300           CONTINUE

               l1 = l1 + arg_incr(T_AXIS, ARG1)
 400        CONTINUE

            m1 = m1 + arg_incr(E_AXIS, ARG1)
 500     CONTINUE

         n1 = n1 + arg_incr(F_AXIS, ARG1)
 600  CONTINUE

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END
