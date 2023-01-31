*
* date1900.F
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
* Steve Hankin
* Dac, 2000
*  5/2007 *acm* Fixing bug 1510. Return the result from date_decode as
*               an argument rather than a return from function; under
*               64-bit the return always yielded 0.
* V69+ *acm* 9/14 Make DATE1900 accept an array of date strings, 
*                 returning an array of coordinates
* V76  *acm* 5/20 Let it work on an array in any grid direction
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
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


      SUBROUTINE date1900_init (id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CHARACTER*120 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Convert date strings into Julian days since 1-Jan-1900 ',
     .        '(integer days using Gregorian calendar)')
      CALL ef_set_desc(id, fcn_desc)  

      CALL ef_set_num_args(id, 1)

      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, YES, NO, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 0)


      arg = 1
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      CALL ef_set_arg_name (id, arg, 'formatted date')
      CALL ef_set_arg_desc(id, arg,
     .         'dd-MMM-yy or dd-MMM-yyyy  (e.g. "20-Dec-00")')
      CALL ef_set_axis_influence_6d(id, arg,
     .                               YES, YES, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE date1900_compute (id, arg_1, result)

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
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER slen, i, j, k, l, m, n
      CHARACTER str_date*12
      REAL res
      INTEGER TM_LENSTR1

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

C  result and arg_1 have same array dimension limits

      DO n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      DO k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
      DO j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      DO i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         CALL ef_get_string_arg_element_6d(id, ARG1, arg_1,
     .                       i, j, k, l, m, n, slen, str_date)

* make it a null-terminated string
         slen = MIN(TM_LENSTR1(str_date)+1,12)
         str_date(slen:slen) = CHAR(0)

         CALL DATE_DECODE(str_date, res)
         IF ( res .EQ. -1E34 ) THEN
            result(i, j, k, l, m, n) = bad_flag_result
         ELSE
            result(i, j, k, l, m, n) = res
         ENDIF

      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END

