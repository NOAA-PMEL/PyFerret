      SUBROUTINE RR_wuz_Qiao_init(id)
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id
      CALL ef_set_desc             (id,'w du/dz from Qiao (1997)')
      CALL ef_set_num_args         (id,2)
      CALL ef_set_axis_inheritance (id,IMPLIED_BY_ARGS, 
     . IMPLIED_BY_ARGS,IMPLIED_BY_ARGS,IMPLIED_BY_ARGS)
      CALL ef_set_arg_name         (id,ARG1,'w') 
      CALL ef_set_arg_desc         (id,ARG1,'w-computed')
      CALL ef_set_arg_unit         (id,ARG1,'cm/sec')
      CALL ef_set_arg_type         (id,ARG1,FLOAT_ARG)
      CALL ef_set_axis_influence   (id,ARG1,YES,YES,YES,YES)
      CALL ef_set_arg_name         (id,ARG2,'u4')
      CALL ef_set_arg_desc         (id,ARG2,'u-station 4')
      CALL ef_set_arg_unit         (id,ARG2,'cm/sec')
      CALL ef_set_arg_type         (id,ARG2,FLOAT_ARG)
      CALL ef_set_axis_influence   (id,ARG2,YES,YES,YES,YES)
      END


      SUBROUTINE RR_wuz_Qiao_compute(id,arg_1,arg_2,result)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      REAL dz,arg_1 (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .               mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL    arg_2 (mem2lox:mem2hix, mem2loy:mem2hiy, 
     .               mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL    result(memreslox:memreshix,memresloy:memreshiy,
     .               memresloz:memreshiz,memreslot:memreshit)
      INTEGER res_lo_ss(4),res_hi_ss(4),res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS),arg_hi_ss(4,EF_MAX_ARGS)
      INTEGER arg_incr(4,EF_MAX_ARGS)
      INTEGER i,j,k,l,i1,j1,k1,l1,i2,j2,k2,l2,iZ,id
      REAL*8  zW(1024)

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_coordinates    (id, ARG1, Z_AXIS, 
     .              arg_lo_ss(Z_AXIS,ARG1),arg_hi_ss(Z_AXIS,ARG1), zW)
      i  = res_lo_ss(X_AXIS)
      i1 = arg_lo_ss(X_AXIS,ARG1) 
      i2 = arg_lo_ss(X_AXIS,ARG2) 
      j  = res_lo_ss(Y_AXIS)
      j1 = arg_lo_ss(Y_AXIS,ARG1) 
      j2 = arg_lo_ss(Y_AXIS,ARG2) 
      l1 = arg_lo_ss(T_AXIS,ARG1) 
      l2 = arg_lo_ss(T_AXIS,ARG2) 
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         iZ = 1
         k1 = arg_lo_ss(Z_AXIS,ARG1)
         k2 = arg_lo_ss(Z_AXIS,ARG2)
         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

           !  ARG_1 = w     ARG_2 = u4

           IF(k.EQ.res_lo_ss(Z_AXIS))THEN
              dz = 100. * (zW(1)+zW(2))/2.
              result(i,j,k,l) = ARG_1(i1,j1,k1,l1)/2.
     .                   *(ARG_2(i2,j2,k2,l2)-
     .                        ARG_2(i2,j2,k2+1,l2))/dz
           ELSEIF(k.EQ. res_hi_ss(Z_AXIS))THEN  ! last z point
              result(i,j,k,l) =  result(i,j,k-1,l)
           ELSE
              dz = 100. * (zW(iZ+1)-zW(iZ-1))/2.
              result(i,j,k,l) = (ARG_1(i1,j1,k1,l1) +
     .                            ARG_1(i1,j1,k1+1,l1))/2
     .                   *(ARG_2(i2,j2,k2,l2)-
     .                        ARG_2(i2,j2,k2+1,l2))/dz    
           ENDIF
           iZ = iZ + 1
           k1 = k1 + arg_incr(Z_AXIS,ARG1)
           k2 = k2 + arg_incr(Z_AXIS,ARG2)
         ENDDO 
         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
      ENDDO               
      END				






