      SUBROUTINE EF_GET_ONE_VAL_SUB(arg_num, memory, mr_list, 
     .	cx_list, value)
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
c* Currently, no checking is done.
c*

      include 'ferret.parm'
      include 'xvariables.cmn'
      include 'xmem_subsc.cmn'

      REAL    memory(mem_blk_size, max_mem_blks)
      REAL    value

      INTEGER arg_num, mr_list(*), cx_list(*)
      INTEGER cx, mr, status

      cx = cx_list(arg_num)
      mr = mr_list(arg_num)
      call ef_get_one_val_sub_sub(memory(1, mr_blk1(mr)), 
     .     lo_ss(arg_num,1), hi_ss(arg_num, 1),
     .     lo_ss(arg_num,2), hi_ss(arg_num, 2),
     .     lo_ss(arg_num,3), hi_ss(arg_num, 3),
     .     lo_ss(arg_num,4), hi_ss(arg_num, 4),
     .     value, status)
      
      RETURN 
      END
