*
*  neutral.F
*
*  John Donners
*  Jun 19th 2002
*
*  Returns neutral density
*
* On a SUN Solaris system compile with:
*
* f77 -PIC -G -z text -z muldefs neutral.F gamma.a /opt/SUNWspro/lib/libM77.so
* /opt/SUNWspro/lib/lib{f77compat,fsu}.so -o neutral.so
*
* The gamma library (gamma.a) should be compiled with the -KPIC option.
* Make sure all libraries are 32 bits compiled (without the -xarch=v9 flag)
* Make sure the file gamma.nc is with the exact path in read-nc.F
* Changed code in gamma-n.f. Now also complains about out of oceanographic
* range, but doesn't quit anymore. Instead now returns -99.2, and jumps to
* the end of the subroutine
* The gamma library can be found at:
* ftp://ftp.marine.csiro.au/pub/jackett/gamma.tar.Z
*
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


      SUBROUTINE neutral_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,' neutral(salinity,temperature) returns'//
     .  ' neutral density' )

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'S')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'T')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      CALL ef_set_num_work_arrays(id,3)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END

      SUBROUTINE neutral_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mz1, mz2, mz3

      INTEGER iwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate double the dimension of the input arguments for work arrays
*  which will be REAL*8

      mz1 = 1 + ABS(arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1))
      mz2 = 1 + ABS(arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1))
      mz3 = 1 + ABS(arg_hi_ss(Z_AXIS,ARG1) - arg_lo_ss(Z_AXIS,ARG1))


*  lon
      iwork = 1
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1,
     .   2*mz1, 1, 1, 1)

*  lat
      iwork = 2
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1,
     .   1, 2*mz2, 1, 1)

*  z
      iwork = 3
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1,
     .   1, 1, 2*mz3, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
*  In this subroutine we compute the result
*
      SUBROUTINE neutral_compute(id, arg_1, arg_2, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .     memresloz:memreshiz, memreslot:memreshit)

      REAL*8 lon(wrk1lox:wrk1hix/2,wrk1loy:wrk1hiy,wrk1loz:wrk1hiz,
     .  wrk1lot:wrk1hit)
      REAL*8 lat(wrk2lox:wrk2hix,wrk2loy:wrk2hiy/2,wrk2loz:wrk2hiz,
     .  wrk2lot:wrk2hit)
      REAL*8 depth(wrk3lox:wrk3hix,wrk3loy:wrk3hiy,wrk3loz:wrk3hiz/2,
     .  wrk3lot:wrk3hit)

* After initialization, the 'res_' arrays contain indexing information
* for the result axes.  The 'arg_' arrays will contain the indexing
* information for each variable's axes.

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER ilon, jlat, kdepth
      REAL*8 dum1,dum2, res

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG1, X_AXIS,
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), lon)
      CALL ef_get_coordinates(id, ARG1, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), lat)
      CALL ef_get_coordinates(id, ARG1, Z_AXIS,
     .   arg_lo_ss(Z_AXIS, ARG1), arg_hi_ss(Z_AXIS, ARG1), depth)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      ilon = 1
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         jlat = 1
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            kdepth = 1
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)


                  IF ( arg_1(i1,j1,k1,l1) .EQ. bad_flag(1) .OR.
     .                 arg_2(i2,j2,k2,l2) .EQ. bad_flag(2) ) THEN

                     result(i,j,k,l) = bad_flag_result

                  ELSE

                     call gamma_n(dble(arg_1(i1,j1,k1,l1)),
     .                 dble(arg_2(i2,j2,k2,l2)),depth(1,1,kdepth,1),1,
     .                 lon(ilon,1,1,1),lat(1,jlat,1,1),res,dum1,dum2)
c                     print*,'s,t,p,lon,lat,res=',
c     .                 dble(arg_1(i1,j1,k1,l1)),
c     .                 dble(arg_2(i2,j2,k2,l2)),depth(1,1,kdepth,1),
c     .                 lon(ilon,1,1,1),lat(1,jlat,1,1),res
                     result(i,j,k,l)=real(res)
                  END IF


                  l1 = l1 + arg_incr(T_AXIS,ARG1)
                  l2 = l2 + arg_incr(T_AXIS,ARG2)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
               kdepth = kdepth + 1
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            jlat = jlat + 1
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         ilon = ilon + 1
 400  CONTINUE


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
