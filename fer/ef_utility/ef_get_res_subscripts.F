c*
c* ef_get_res_subscripts.F
c*
c* Jonathan Callahan
c* Apr 6th 1998
c*
c*

c*
c* Initialize the result context limits the EF will use.  
c*

      SUBROUTINE EF_GET_RES_SUBSCRIPTS(id, res_lo_ss,
     .     res_hi_ss, res_incr)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xvariables.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER idim

      INTEGER mres

c* The memory resident id for the rsult, "mres", was cached by 
c* Ferret and must be retrieved to be accessible to the external function.
      CALL EF_GET_MRES(mres)

      DO 100 idim=1,4

         res_lo_ss(idim) = mr_lo_ss(mres, idim)
         res_hi_ss(idim) = mr_hi_ss(mres, idim)
         IF (res_lo_ss(idim) .EQ. res_hi_ss(idim)) 
     .        THEN
            res_incr(idim) = 0
         ELSE
            res_incr(idim) = 1
         ENDIF

 100  CONTINUE

      RETURN 
      END
