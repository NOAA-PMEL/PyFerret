*
* scat2ddups.F
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
* Jan, 2001

* This function converts a Julian day from 1-jan-1900 to a trio of 3 values
* on the Z axis: year, month, day

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


      SUBROUTINE scat2ddups_init (id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc (id,
     .   'Flag non-unique points. '
     .   //'L=1 - # dups of this pt, 2-sequence number' )  

      CALL ef_set_num_args(id, 4)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .                                 IMPLIED_BY_ARGS, ABSTRACT)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 0)


      arg = 1
      CALL ef_set_arg_name (id, arg, 'coord 1')
      CALL ef_set_arg_unit(id, arg, 'none')
      CALL ef_set_arg_desc(id, arg,
     .         'Coordinate 1 (normally longitude)')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, NO)

      arg = 2
      CALL ef_set_arg_name (id, arg, 'coord 2')
      CALL ef_set_arg_unit(id, arg, 'none')
      CALL ef_set_arg_desc(id, arg,
     .         'Coordinate 2 (normally latitude)')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, NO)

      arg = 3
      CALL ef_set_arg_name (id, arg, 'epsilon 1')
      CALL ef_set_arg_unit(id, arg, 'none')
      CALL ef_set_arg_desc(id, arg,
     .         'Two coord 1s within epsilon 1 considered duplicates')
      CALL ef_set_axis_influence (id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name (id, arg, 'epsilon 2')
      CALL ef_set_arg_unit(id, arg, 'none')
      CALL ef_set_arg_desc(id, arg,
     .         'Two coord 2s within epsilon 2 considered duplicates')
      CALL ef_set_axis_influence (id, arg, NO, NO, NO, NO)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

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

      SUBROUTINE scat2ddups_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

* 3 output values: year, month, day 
        CALL EF_SET_AXIS_LIMITS(id, T_AXIS, 1, 2)
        
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



*
* In this subroutine we compute the result
*
      SUBROUTINE scat2ddups_compute (id, arg_1, arg_2, arg_3, arg_4,
     .                               result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3
      REAL arg_4

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)
 

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
        INTEGER ii, size1, size2

* get the subscripting limits and flags
        CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
        CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
        CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* check to make sure that we were not passes a span along the T axis
        IF ( (arg_lo_ss(T_AXIS,ARG1) .NE.  arg_hi_ss(T_AXIS,ARG1))
     .  .OR. (arg_lo_ss(T_AXIS,ARG2) .NE.  arg_hi_ss(T_AXIS,ARG2)) )
     .          CALL EF_BAIL_OUT(id,
     .                  'Cannot handle a T range on argument') 

* ensure that the sizes of the inputs match
        size1 = 1
        size2 = 1
        DO ii = 1,4
           size1 = size1 * (arg_hi_ss(ii,ARG1)-arg_lo_ss(ii,ARG1)+1)
           size2 = size2 * (arg_hi_ss(ii,ARG2)-arg_lo_ss(ii,ARG2)+1)
        END DO
        IF (size1 .NE. size2)  CALL EF_BAIL_OUT(id,
     .                  'Coordinate arrays are not conformable')

* ensure that the epsilon values make sense
        IF (arg_3.LT.0.0 .OR. arg_4.LT.0.0) CALL EF_BAIL_OUT(id,
     .                  'Negative epsilon value')

* flag the duplicates
        CALL FLAG2DDUPS(size1,
     .                  arg_1(arg_lo_ss(X_AXIS,ARG1),
     .                        arg_lo_ss(Y_AXIS,ARG1),
     .                        arg_lo_ss(Z_AXIS,ARG1),
     .                        arg_lo_ss(T_AXIS,ARG1)),
     .                  arg_2(arg_lo_ss(X_AXIS,ARG2),
     .                        arg_lo_ss(Y_AXIS,ARG2),
     .                        arg_lo_ss(Z_AXIS,ARG2),
     .                        arg_lo_ss(T_AXIS,ARG2)),
     .                  arg_3,
     .                  arg_4,
     .                  result(memreslox, memresloy, 
     .                         memresloz, 1),
     .                  result(memreslox, memresloy, 
     .                         memresloz, 2)  )
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END

