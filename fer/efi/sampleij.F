*
* sampleij.F
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
* Ansley Manke
* APR 1999
*
* This function samples 4-d data on the x and y axes indicated by args 2 and 3
* 4/5/99  Ansley Manke 
*         Result is abstract on the x axis, normal on the y axis,
*         and keeps the z and t axes of the input 4-d data.
* 4/9/09  The function never tested for the bad-flag of the input variable.
* 1/29/16 Ticketn2358: Clarify the function description and argument descriptions
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


      SUBROUTINE sampleij_init(id)

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
     .  ('Returns data sampled by matching coordinates at a subset of ',
     .   'grid points, defined by (XPTS, YPTS)')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT,        NORMAL,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 2)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'variable (x,y,z,t,e,f) to sample')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 'X coordinates of grid points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 'Y coordinates of grid points')
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

      SUBROUTINE sampleij_result_limits(id)

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
      INTEGER nx

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1

      CALL ef_set_axis_limits(id, X_AXIS, 1, nx)

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
      SUBROUTINE sampleij_work_size(id)

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
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*
      INTEGER mxl, myl, mxh, myh
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mxl = arg_lo_ss(X_AXIS,ARG1)
      myl = arg_lo_ss(Y_AXIS,ARG1)

*  Allocate double the length of the axes for REAL*8 work arrays.
      mxh = arg_hi_ss(X_AXIS,ARG1) * 2
      myh = arg_hi_ss(Y_AXIS,ARG1) * 2

*  xaxdat
      CALL ef_set_work_array_dims_6d(id, 1,
     .                              mxl, 1, 1, 1, 1, 1,
     .                              mxh, 1, 1, 1, 1, 1)

*  yaxdat
      CALL ef_set_work_array_dims_6d(id, 2,
     .                              myl, 1, 1, 1, 1, 1,
     .                              myh, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE sampleij_compute(id, arg_1, arg_2, arg_3, result, 
     .       xaxdat, yaxdat)

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

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

      REAL*8 xaxdat(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .              wrk1loz:wrk1hiz, wrk1lot:wrk1hit,
     .              wrk1loe:wrk1hie, wrk1lof:wrk1hif)
      REAL*8 yaxdat(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .              wrk2loz:wrk2hiz, wrk2lot:wrk2hit,
     .              wrk2loe:wrk2hie, wrk2lof:wrk2hif)

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

      LOGICAL TM_FPEQ, its_dsg

      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER imatch, jmatch
      REAL xgrid, ygrid, a
      CHARACTER*255 err_msg

* Does the variable come from a Discrete Sampling Geometries dataset?  

      CALL EF_GET_ITS_DSG(id, ARG1, its_dsg )
      IF (its_dsg) THEN
         err_msg = 
     . 'Data is from a Discrete Sampling Geometries (DSG) dataset. Function does not apply.'
       GOTO 999
      ENDIF

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      IF ((arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2)) .GT. 0  .OR.
     .    (arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2)) .GT. 0  .OR.
     .    (arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2)) .GT. 0  .OR.
     .    (arg_hi_ss(E_AXIS,ARG2) - arg_lo_ss(E_AXIS,ARG2)) .GT. 0  .OR.
     .    (arg_hi_ss(F_AXIS,ARG2) - arg_lo_ss(F_AXIS,ARG2)) .GT. 0) THEN
         WRITE (err_msg, 10)
         GO TO 999
      ENDIF

      IF ((arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3)) .GT. 0  .OR.
     .    (arg_hi_ss(Z_AXIS,ARG3) - arg_lo_ss(Z_AXIS,ARG3)) .GT. 0  .OR.
     .    (arg_hi_ss(T_AXIS,ARG3) - arg_lo_ss(T_AXIS,ARG3)) .GT. 0  .OR.
     .    (arg_hi_ss(E_AXIS,ARG3) - arg_lo_ss(E_AXIS,ARG3)) .GT. 0  .OR.
     .    (arg_hi_ss(F_AXIS,ARG3) - arg_lo_ss(F_AXIS,ARG3)) .GT. 0) THEN
         WRITE (err_msg, 20)
         GO TO 999
      ENDIF

   10 FORMAT ('SAMPLEIJ expects Argument 2 to be a list of X ',
     .       'coordinates whose ''grid'' consists of an X axis only')
   20 FORMAT ('SAMPLEIJ expects Argument 3 to be a list of Y ',
     .       'coordinates whose ''grid'' consists of an X axis only')
         

*  Get x and y coordinates of the data array.

      CALL ef_get_coordinates(id, ARG1, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xaxdat)
      CALL ef_get_coordinates(id, ARG1, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yaxdat)

      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      m2 = arg_lo_ss(E_AXIS,ARG2)
      n2 = arg_lo_ss(F_AXIS,ARG2)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      m3 = arg_lo_ss(E_AXIS,ARG3)
      n3 = arg_lo_ss(F_AXIS,ARG3)

*  For each (xpt,ypt) pair, search the data array arg_1 and match its (x,y)
*  coordinates.  Keep the result in result(i_point,_,k,l,m,n) for all 
*  k,l,m,n in the data array.

      j = res_lo_ss(Y_AXIS)
      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         imatch = ef_unspecified_int4
         jmatch = ef_unspecified_int4

         IF ( arg_2(i2,j2,k2,l2,m2,n2) .NE. bad_flag(ARG2) .AND.
     .        arg_3(i3,j3,k3,l3,m3,n3) .NE. bad_flag(ARG3) ) THEN

            DO 110 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1) 
               xgrid = xaxdat(i1,1,1,1,1,1)
               IF ( TM_FPEQ(xgrid, arg_2(i2,j2,k2,l2,m2,n2)) ) THEN
                   imatch = i1
               ENDIF
  110       CONTINUE

            DO 120 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1) 
               ygrid = yaxdat(j1,1,1,1,1,1)
               IF ( TM_FPEQ(ygrid, arg_3(i3,j3,k3,l3,m3,n3)) ) THEN
                  jmatch = j1
               ENDIF
  120       CONTINUE

         ENDIF

         IF ( (imatch .NE. ef_unspecified_int4) .AND.
     .        (jmatch .NE. ef_unspecified_int4) ) THEN

            n1 = arg_lo_ss(F_AXIS,ARG1)
            DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

            m1 = arg_lo_ss(E_AXIS,ARG1)
            DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               k1 = arg_lo_ss(Z_AXIS,ARG1)
               DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  a = arg_1(imatch,jmatch,k1,l1,m1,n1)
                  IF ( a .NE. bad_flag(ARG1) ) THEN
                     result(i,j,k,l,m,n) = 
     .                     arg_1(imatch,jmatch,k1,l1,m1,n1)
                  ELSE
                     result(i,j,k,l,m,n) = bad_flag_result
                  ENDIF

                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300           CONTINUE

               l1 = l1 + arg_incr(T_AXIS,ARG1)
 400        CONTINUE

               m1 = m1 + arg_incr(E_AXIS,ARG1)
 500        CONTINUE

               n1 = n1 + arg_incr(F_AXIS,ARG1)
 600        CONTINUE

         ELSE

*  Data coordinates do not match (xpts,ypts) coordinates

            n1 = arg_lo_ss(F_AXIS,ARG1)
            DO 650 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

            m1 = arg_lo_ss(E_AXIS,ARG1)
            DO 550 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            DO 450 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               k1 = arg_lo_ss(Z_AXIS,ARG1)
               DO 350 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

                  result(i,j,k,l,m,n) = bad_flag_result

                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
 350           CONTINUE

               l1 = l1 + arg_incr(T_AXIS,ARG1)
 450        CONTINUE

               m1 = m1 + arg_incr(E_AXIS,ARG1)
 550        CONTINUE

               n1 = n1 + arg_incr(F_AXIS,ARG1)
 650        CONTINUE

         ENDIF


         i2 = i2 + arg_incr(X_AXIS,ARG2)
         i3 = i3 + arg_incr(X_AXIS,ARG3)
 100  CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)
      RETURN

      END
