c     
c     efcn_get_workspace_addr.F
c     
c     Jonathan Callahan
c     September 1998
c     

c     
c     This subroutine is called by efcn_compute_ inside EF_ExternalUtil.c. 
c     When workspace is requested by an external function it is created 
c     inside of efcn_compute_ but efcn_compute_ doesn't know how to access
c     variuos Ferret common areas.  (And I don't want to bother with
c     the C/Fortran column major/row major stuff).  This subroutine accesses
c     the common areas and passes back the appropriate address to efcn_compute_.
c     

      SUBROUTINE EFCN_GET_WORKSPACE_ADDR(memory, work_blk, workspace)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xvariables.cmn'

      INTEGER work_blk
      REAL memory( mem_blk_size, max_mem_blks ), workspace

      workspace = memory(work_blk, 1)

      RETURN 
      END
