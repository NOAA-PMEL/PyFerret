*
* picky.F
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
* March 1999
*  Redo July 2000
*
*  Apply a mask to data.  Returns a result with an ABSTRACT Y axis.
*  Take only those points where mask had valid (non-missing) points.
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


      SUBROUTINE picky_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data masked; result on abstract y axis ',
     .        'where mask has valid points')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id,  IMPLIED_BY_ARGS, ABSTRACT, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)


      arg = 1
      CALL ef_set_arg_name(id, arg, 'VAR')
      CALL ef_set_arg_desc(id, arg, 'data to mask')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'MASK')
      CALL ef_set_arg_desc(id, arg, 
     .        '1-D mask, same length as y-axis of VAR')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
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

      SUBROUTINE picky_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      INTEGER my_lo_l, my_hi_l

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*     Set the abstract Y axis.
*

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1

      my_lo_l = 1
      my_hi_l = ny
      IF (arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) .EQ. 0)
     .    my_hi_l = max(nx,ny,nz,nt)

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
      SUBROUTINE picky_compute(id, arg_1, arg_2, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
     .     
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .      memresloz:memreshiz, memreslot:memreshit)

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
      INTEGER id, i, j, k, l
      INTEGER i1, j1, k1, l1
      INTEGER j2, k2, l2
      INTEGER nlen1, nlen2, ndims
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check that ARG2 is 1-D.  

      ndims = 0
      DO 88 m = X_AXIS, T_AXIS
         IF (arg_incr(m,ARG2) .GT. 0) THEN
           nlen2 = arg_hi_ss(m,ARG2) - arg_lo_ss(m,ARG2) 
           ndims = ndims + 1
         ENDIF
  88  CONTINUE

      IF (ndims .LE. 0  .OR.  ndims .GT. 1) THEN
         WRITE (errtxt, *) 'pickY: ARG2 (mask) must be 1-dimensional'
         GOTO 999
      ENDIF

C  Check that the dimension of the mask matches the y axis of the variable.

      IF (arg_incr(Y_AXIS,ARG1) .GT. 0) nlen1 = arg_hi_ss(Y_AXIS,ARG1) 
     .    - arg_lo_ss(Y_AXIS,ARG1)

      IF (nlen2 .NE. nlen1) THEN
         WRITE (errtxt, *) 
     .    'ARG2 (mask) must be the same length as the y-axis of ARG1'
         GOTO 999
      ENDIF

C  Apply the mask.

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 400 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         k2 = arg_lo_ss(Z_AXIS,ARG2)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            DO 200 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                j = res_lo_ss(Y_AXIS)
                j1 = arg_lo_ss(Y_AXIS,ARG1)
                j2 = arg_lo_ss(Y_AXIS,ARG2)

                DO 100 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

*  Check that this value of ARG_2 is not missing; set the result

                   If (arg_2(i2,j2,k2,l2) .ne. bad_flag(ARG2)) THEN
                        result(i,j,k,l) = arg_1(i1,j1,k1,l1)
                  ELSE
                        result(i,j,k,l) = bad_flag_result
                  END IF

                  j1 = j1 + arg_incr(Y_AXIS,ARG1) 
                  j2 = j2 + arg_incr(Y_AXIS,ARG2) 

 100           CONTINUE

               l1 = l1 + arg_incr(T_AXIS,ARG1)
               l2 = l2 + arg_incr(T_AXIS,ARG2)
 200        CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
            k2 = k2 + arg_incr(Z_AXIS,ARG2)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
 400  CONTINUE

      RETURN 

999   CONTINUE

      CALL EF_BAIL_OUT(id, errtxt)


      END
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************
