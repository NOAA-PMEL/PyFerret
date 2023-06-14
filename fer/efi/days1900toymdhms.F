*
* days1900toymdhms.F
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
*
* 12/12/02 *acm* bug fix: define the REAL*8 parameter  psecsto1900 using a 
*                double precision constant .D0; the constant lost precision
*                under linux.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

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


      SUBROUTINE days1900toymdhms_init (id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc (id,
     .   'Convert Julian day to values y,m,d,h,m,s on k=1,...,6' )

      CALL ef_set_num_args(id, 1)

      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                ABSTRACT,        IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 0)


      arg = 1
      CALL ef_set_arg_name (id, arg, 'day1900')
      CALL ef_set_arg_unit(id, arg, 'none')
      CALL ef_set_arg_desc(id, arg,
     .         'Julian day counted from 1-jan-1900')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, NO, YES, YES, YES)

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

      SUBROUTINE days1900toymdhms_result_limits(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

* 3 output values: year, month, day 
        CALL EF_SET_AXIS_LIMITS(id, Z_AXIS, 1, 6)
        
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



*
* In this subroutine we compute the result
*
      SUBROUTINE days1900toymdhms_compute (id, arg_1, result)

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

        INTEGER i, j, k, l, m, n, k1, yr, mon, day, hr, min
        REAL sec
        REAL*8 num_secs

        REAL*8  psecsto1900  ! seconds from 1-jan-0000 to 1-jan-1900
        PARAMETER ( psecsto1900 = 59958230400.D0 )

* get the subscripting limits and flags
        CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
        CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
        CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* check to make sure that we were not passes a span along the Z axis
        IF (arg_lo_ss(Z_AXIS,1) .NE.  arg_hi_ss(Z_AXIS,1)) CALL
     .          EF_BAIL_OUT(id, 'Cannot handle a Z range on argument') 

* loop over all of the input values
* assumes input and result array dimension limits are the same
        k1 = arg_lo_ss(Z_AXIS,1)
        DO 600 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
          DO 500 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
              DO 200 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
                DO 100 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                  IF ( arg_1(i,j,k1,l,m,n) .EQ. bad_flag(1) ) THEN

                    DO k = 1, 6
                       result(i,j,k,l,m,n) = bad_flag_result
                    ENDDO

                  ELSE

                    num_secs = 24.D0*60.D0*60.D0 * arg_1(i,j,k1,l,m,n)
     .                       + psecsto1900
                    CALL SECS_TO_YMDHMS( num_secs,
     .                                   yr, mon, day, hr, min, sec )
                    result(i,j,1,l,m,n) = FLOAT(yr)
                    result(i,j,2,l,m,n) = FLOAT(mon)
                    result(i,j,3,l,m,n) = FLOAT(day)
                    result(i,j,4,l,m,n) = FLOAT(hr)
                    result(i,j,5,l,m,n) = FLOAT(min)
                    result(i,j,6,l,m,n) = sec

                 END IF

 100           CONTINUE
 200         CONTINUE
 400       CONTINUE
 500     CONTINUE
 600   CONTINUE


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END

