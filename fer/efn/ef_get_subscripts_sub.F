c*
c* ef_get_subscripts_sub.F
c*
c* Jonathan Callahan
c* Feb 19th 1998
c*
c*

c*
c* This subroutine is called by code inside EF_ExternalUtil.c. It will
c* initialize the context limits the EF will use.  
c*

      SUBROUTINE EF_GET_SUBSCRIPTS_SUB(cx_list, ss_lo_lim,
     .     ss_hi_lim, ss_incr)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xcontext.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER cx_list(EF_MAX_ARGS)
      INTEGER ss_lo_lim(4,EF_MAX_ARGS), ss_hi_lim(4,EF_MAX_ARGS),
     .     ss_incr(4,EF_MAX_ARGS)

      INTEGER idim, iarg

      DO 200 iarg=1,EF_MAX_ARGS
         DO 100 idim=1,4

            ss_lo_lim(idim,iarg) = cx_lo_ss(cx_list(iarg), idim)
            ss_hi_lim(idim,iarg) = cx_hi_ss(cx_list(iarg), idim)
            IF (ss_lo_lim(idim,iarg) .EQ. ss_hi_lim(idim,iarg)) 
     .           THEN
               ss_incr(idim, iarg) = 0
            ELSE
               ss_incr(idim, iarg) = 1
            ENDIF

 100     CONTINUE
 200  CONTINUE

      RETURN 
      END
