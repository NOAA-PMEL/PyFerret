c*
c* ef_get_one_val_sub.F
c*
c* Jonathan Callahan
c* Dec 17th 1997
c*
c*

c*
c* This subroutine is called by an external function which wishes
c* to get the value of a 1x1x1x1 variable.
c*
c* JC_TODO: Currently, no checking is done.
c*

      SUBROUTINE EF_GET_ONE_VAL_SUB(memory, iarg, value)

      include 'ferret_cmn/ferret.parm'
      include 'ferret_cmn/xvariables.cmn'
      include 'ferret_cmn/xmem_subsc.cmn'
      include 'ferret_cmn/EF_Util.cmn'

      REAL    value

      INTEGER iarg
      INTEGER mr, status

      REAL    memory(mem_blk_size, max_mem_blks)

      INTEGER mr_list(EF_MAX_ARGS)

c* The following variables were cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_MR_LIST(mr_list)

      mr = mr_list(iarg)
      CALL EF_GET_ONE_VAL_SUB_SUB(memory(1, mr_blk1(mr)), 
     .     lo_ss(iarg,1), hi_ss(iarg, 1),
     .     lo_ss(iarg,2), hi_ss(iarg, 2),
     .     lo_ss(iarg,3), hi_ss(iarg, 3),
     .     lo_ss(iarg,4), hi_ss(iarg, 4),
     .     value, status)
      
      RETURN 
      END
