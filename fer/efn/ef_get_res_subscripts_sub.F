c*
c* ef_get_res_subscripts_sub.F
c*
c* Jonathan Callahan
c* Apr 6th 1998
c*
c*

c*
c* This subroutine is called by code inside EF_ExternalUtil.c. It will
c* initialize the result context limits the EF will use.  
c*

      SUBROUTINE EF_GET_RES_SUBSCRIPTS_SUB(mres, res_lo_ss,
     .     res_hi_ss, res_incr)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xvariables.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER mres
      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

      INTEGER idim

      DO 100 idim=1,4

            res_lo_ss(idim) = mr_lo_ss(mres, idim)
            res_hi_ss(idim) = mr_hi_ss(mres, idim)
            IF (res_lo_ss(idim) .EQ. res_hi_ss(idim)) 
     .           THEN
               res_incr(idim) = 0
            ELSE
               res_incr(idim) = 1
            ENDIF

 100  CONTINUE

      RETURN 
      END
