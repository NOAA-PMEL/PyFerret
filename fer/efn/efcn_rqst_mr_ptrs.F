c     
c     efcn_rqst_mr_ptrs.F
c     
c     Steve hankin
c     19 Jan., 2017
c     

c     
c     This subroutine is called by efcn_compute_ inside EF_ExternalUtil.c. 
c     It passes the c pointers for the function arguments and result 
c     from their storage in FORTRAN90 pointers back to c

        SUBROUTINE EFCN_RQST_MR_PTRS(nargs, mr_list, mr_res)

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	nargs, mr_list(nargs), mr_res

* internal variable declarations
        INTEGER i

* pass the arg pointers to c
        DO i = 1, nargs
           CALL EFCN_PASS_ARG_PTR(i, memry(mr_list(i))%ptr )
        ENDDO

* pass the result pointer to c
        CALL EFCN_PASS_RES_PTR(memry(mr_res)%ptr)

        RETURN 
        END
