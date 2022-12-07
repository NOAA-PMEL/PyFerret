*
* sampleyz.F
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without anx fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and anx derivative works thereof, and
*  its supporting documentation for anx purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in anx publications that result from
*  the use of this software or in anx product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in anx advertising or publicity to endorse or promote anx products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with anx support, consulting,
*  training or assistance of anx kind with regard to the use, operation
*  and performance of this software nor to provide the user with anx
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND Anx EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR Anx SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR Anx DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Ansley Manke
* From samplexy.F
* Wednesday, May 02, 2001
* 11-Jan-06 *acm* declare ylo, yhi, zlo, zhi as integer not real
*
* V68   3/12 Fix arg description.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* This function samples 4-d data at y and z pts indicated by args 2 and 3
* 
*         Result is abstract on the Y axis, normal on the z axis,
*         and keeps the x and t axes of the input 4-d data.

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


      SUBROUTINE sampleyz_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled at a set of (Y,Z) points, ',
     .        'using linear interpolation')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, ABSTRACT,
     .                                NORMAL,          IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 2)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'variable (x,y,z,t,e,f) to sample')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, NO, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 'Y values of sample points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 'Z values of sample points')
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

      SUBROUTINE sampleyz_result_limits(id)

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
      INTEGER nx, ny, nz, nt, ne, nf
      INTEGER my_lo, my_hi

*     Use utility functions to get context information about the 
*     second argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1
      ne = arg_hi_ss(E_AXIS, ARG2) - arg_lo_ss(E_AXIS, ARG2) + 1
      nf = arg_hi_ss(F_AXIS, ARG2) - arg_lo_ss(F_AXIS, ARG2) + 1

      my_lo = 1
      my_hi = MAX(nx,ny,nz,nt,ne,nf)

      CALL ef_set_axis_limits(id, Y_AXIS, my_lo, my_hi)

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
      SUBROUTINE sampleyz_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

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

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER myl, myh, mzl, mzh

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate double the length of the axes for REAL*8 work arrays.

      myl = arg_lo_ss(Y_AXIS,ARG1)
      mzl = arg_lo_ss(Z_AXIS,ARG1)

      myh = myl + 
     .     2* (arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1)
      mzh = mzl + 
     .     2* (arg_hi_ss(Z_AXIS,ARG1) - arg_lo_ss(Z_AXIS,ARG1) + 1)

*  yaxdat
      CALL ef_set_work_array_dims_6d(id, 1,
     .                              myl, 1, 1, 1, 1, 1,
     .                              myh, 1, 1, 1, 1, 1)

*  zaxdat
      CALL ef_set_work_array_dims_6d(id, 2,
     .                              mzl, 1, 1, 1, 1, 1,
     .                              mzh, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE sampleyz_compute(id, arg_1, arg_2, arg_3, result,
     .                            yaxdat, zaxdat)

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

* Ignore the Y/Z/T/E/F dimensions in the work arrays since
* they are not used and Fortran is column major
      REAL*8 yaxdat(wrk1lox:wrk1lox+(wrk1hix-wrk1lox)/2)
      REAL*8 zaxdat(wrk2lox:wrk2lox+(wrk2hix-wrk2lox)/2)

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

      INTEGER ny, nyx, nyy, nyz, nyt, nye, nyf
      INTEGER nz, nzx, nzy, nzz, nzt, nze, nzf
      INTEGER ndimy, ndimz
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER jmatch, kmatch
      INTEGER jbot, jtop, kbot, ktop
      INTEGER ylo, yhi, zlo, zhi
      REAL fybot, fytop, fbb, ftb, fbt, ftt
      REAL ybot, ytop, zbot, ztop
      REAL yfrac, zfrac
      REAL delmody, delmodz, ypt, zpt
      CHARACTER*255 err_msg

C  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), modulo(6), regular(6)

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

C Validate argument dimensions
      nyx = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      nyy = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1
      nyz = arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1
      nyt = arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1
      nye = arg_hi_ss(E_AXIS,ARG2) - arg_lo_ss(E_AXIS,ARG2) + 1
      nyf = arg_hi_ss(F_AXIS,ARG2) - arg_lo_ss(F_AXIS,ARG2) + 1
      ny = MAX(nyx, nyy, nyz, nyt, nye, nyf)

      nzx = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      nzy = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      nzz = arg_hi_ss(Z_AXIS,ARG3) - arg_lo_ss(Z_AXIS,ARG3) + 1
      nzt = arg_hi_ss(T_AXIS,ARG3) - arg_lo_ss(T_AXIS,ARG3) + 1
      nze = arg_hi_ss(E_AXIS,ARG3) - arg_lo_ss(E_AXIS,ARG3) + 1
      nzf = arg_hi_ss(F_AXIS,ARG3) - arg_lo_ss(F_AXIS,ARG3) + 1
      nz = MAX(nzx, nzy, nzz, nzt, nze, nzf)

      ndimy = 0
      ndimz = 0
      DO i = X_AXIS,F_AXIS
         IF ( arg_hi_ss(i,ARG2) .GT. arg_lo_ss(i,ARG2) ) 
     .      ndimy = ndimy + 1
         IF ( arg_hi_ss(i,ARG3) .GT. arg_lo_ss(i,ARG3) ) 
     .      ndimz= ndimz + 1
      ENDDO

      IF ( (ny .NE. nz) .OR. (ndimy .GT. 1) .OR. (ndimz .GT.1) ) THEN
         err_msg =
     .  'Arguments 2 and 3 must be 1-dimensional lists of equal length'
         GO TO 999
      ENDIF

*  Get y and z coordinates of the data to be sampled.

      CALL ef_get_coordinates(id, ARG1, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yaxdat)
      CALL ef_get_coordinates(id, ARG1, Z_AXIS,
     .   arg_lo_ss(Z_AXIS, ARG1), arg_hi_ss(Z_AXIS, ARG1), zaxdat)

*  Check to see if input y or z axis is modulo

      CALL ef_get_axis_info_6d(id, ARG1, ax_name, ax_units,
     .                         backward, modulo, regular)

      ylo = arg_lo_ss(Y_AXIS,ARG1)
      yhi = arg_hi_ss(Y_AXIS,ARG1)
      zlo = arg_lo_ss(Z_AXIS,ARG1)
      zhi = arg_hi_ss(Z_AXIS,ARG1)

* Is this enough? (see samplexy and samplexyt)
      IF ( modulo(Y_AXIS) ) THEN
         delmody = yaxdat(yhi) - yaxdat(ylo)
      ENDIF
      IF ( modulo(Z_AXIS) ) THEN
         delmodz = zaxdat(zhi) - zaxdat(zlo)
      ENDIF

*  For each (ypt,zpt) pair, search the data array 
*   arg_1 for the nearest higher (y,z) grid coordinates.  Interpolate 
*   in 2 directions for the result.

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

* Z-axis is normal to the results
      k = res_lo_ss(Z_AXIS)

      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         ypt = arg_2(i2,j2,k2,l2,m2,n2)
         zpt = arg_3(i3,j3,k3,l3,m3,n3)

         IF ( ypt .EQ. bad_flag(ARG2) .OR.
     .        zpt .EQ. bad_flag(ARG3) ) THEN
            DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
            ENDDO
            ENDDO
            ENDDO
            ENDDO

*           Go to the next pair of coordinates
            GOTO 690
         ENDIF

* Try to get ypt in range if the Y-axis is modulo
         IF ( modulo(Y_AXIS) ) THEN
            DO WHILE ( ypt .GE. yaxdat(yhi) ) 
               ypt = ypt - delmody
            ENDDO
            DO WHILE ( ypt .LT. yaxdat(ylo) )
               ypt = ypt + delmody
            ENDDO
         ENDIF

* Check if ypt is in the Y-axis range.
         jbot = ef_unspecified_int4
         jmatch = ef_unspecified_int4

         DO j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
            IF ( TM_FPEQ(ypt, yaxdat(j1)) ) THEN
               jmatch = j1
               GOTO 220
            ELSE IF ( ypt .GT. yaxdat(j1) ) THEN
               jbot = j1
            ENDIF
	 ENDDO

* Check that ypt is not beyond the upper end of the Y-axis range.
         IF ( ypt .GT. yaxdat(yhi) ) THEN
            jbot = ef_unspecified_int4
         ENDIF

  220    IF ( jmatch .NE. ef_unspecified_int4 ) THEN
            jbot = jmatch
            jtop = jmatch
         ELSE IF ( jbot .NE. ef_unspecified_int4 ) THEN
            jtop = jbot + 1
         ELSE
*           ypt outside the grid
            DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
            ENDDO
            ENDDO
            ENDDO
            ENDDO

*           Go to the next pair of coordinates
            GOTO 690
         ENDIF

         ybot = yaxdat(jbot)
         ytop = yaxdat(jtop)
         IF ( ytop .EQ. ybot ) THEN 
            yfrac = 0.
         ELSE
            yfrac = (ypt - ybot) / (ytop - ybot)
         ENDIF

* Try to get zpt in range if the Z-axis is modulo
         IF ( modulo(Z_AXIS) ) THEN
            DO WHILE ( zpt .GE. zaxdat(zhi) ) 
               zpt = zpt - delmodz
            ENDDO
            DO WHILE ( zpt .LT. zaxdat(zlo) )
               zpt = zpt + delmodz
            ENDDO
         ENDIF

* Check if zpt is in the Z-axis range.
         kbot = ef_unspecified_int4
         kmatch = ef_unspecified_int4

         DO k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
            IF ( TM_FPEQ(zpt, zaxdat(k1)) ) THEN
               kmatch = k1
               GOTO 320
            ELSE IF ( zpt .GT. zaxdat(k1) ) THEN
               kbot = k1
            ENDIF
         ENDDO

* Check that zpt is not beyond the upper end of the Z-axis range.
         IF ( zpt .GT. zaxdat(zhi) ) THEN
            kbot = ef_unspecified_int4
         ENDIF

  320    IF ( kmatch .NE. ef_unspecified_int4 ) THEN
            kbot = kmatch
            ktop = kmatch
         ELSE IF ( kbot .NE. ef_unspecified_int4 ) THEN
            ktop = kbot + 1
         ELSE
*           zpt outside the grid
            DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
            ENDDO
            ENDDO
            ENDDO
            ENDDO
*           Go to the next pair of coordinates
            GOTO 690
         ENDIF

         zbot = zaxdat(kbot)
         ztop = zaxdat(ktop)
         IF ( ztop .EQ. zbot ) THEN 
            zfrac = 0.
         ELSE
            zfrac = (zpt - zbot) / (ztop - zbot)
         ENDIF

         n1 = arg_lo_ss(F_AXIS,ARG1)
         DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

         l1 = arg_lo_ss(T_AXIS,ARG1)
         DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1)
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

*  First interpolate in y, getting values of the fcn at (y,kbot) and (y,ktop)
               fbb = arg_1(i1,jbot,kbot,l1,m1,n1)
               ftb = arg_1(i1,jtop,kbot,l1,m1,n1)
               fbt = arg_1(i1,jbot,ktop,l1,m1,n1)
               ftt = arg_1(i1,jtop,ktop,l1,m1,n1)

               IF ( fbb .NE. bad_flag(ARG1)  .AND.
     .              ftb .NE. bad_flag(ARG1)  .AND.
     .              fbt .NE. bad_flag(ARG1)  .AND.
     .              ftt .NE. bad_flag(ARG1) ) THEN

                  fybot = fbb + yfrac * (ftb - fbb)
                  fytop = fbt + yfrac * (ftt - fbt)

*  Now interpolate in z, getting value at (y,z)
                  result(i,j,k,l,m,n) = fybot + zfrac * (fytop - fybot)

               ELSE

*                 Some component was undefined
                  result(i,j,k,l,m,n) = bad_flag_result

               ENDIF

               i1 = i1 + arg_incr(X_AXIS,ARG1)
            ENDDO

            l1 = l1 + arg_incr(T_AXIS,ARG1)
         ENDDO

            m1 = m1 + arg_incr(E_AXIS,ARG1)
         ENDDO

            n1 = n1 + arg_incr(F_AXIS,ARG1)
         ENDDO

* Only one of the ARG2 and one of ARG3 actually increments
 690     i2 = i2 + arg_incr(X_AXIS,ARG2)
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
         k2 = k2 + arg_incr(Z_AXIS,ARG2)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         m2 = m2 + arg_incr(E_AXIS,ARG2)
         n2 = n2 + arg_incr(F_AXIS,ARG2)

         i3 = i3 + arg_incr(X_AXIS,ARG3)
         j3 = j3 + arg_incr(Y_AXIS,ARG3)
         k3 = k3 + arg_incr(Z_AXIS,ARG3)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         m3 = m3 + arg_incr(E_AXIS,ARG3)
         n3 = n3 + arg_incr(F_AXIS,ARG3)

      ENDDO ! j loop

      RETURN

 999  CALL ef_bail_out (id, err_msg)

      END
