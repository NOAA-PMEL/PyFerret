	BLOCK DATA XVARIABLES_DATA

* initialization of COMMON / XVARIABLES /
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/31/86
* .
* .
* .
* revision 2.0 - 7/21/88 - initialize mblk_size/flink, mr_priority/protected
* V200:  5/17/89 - eliminated _dim% --> 4D symmetry for memory resident data
*		 - all cvar-related stuff to XCALC_VARS
* 		 - deleted variables flagged in mr_protected and mr_del_ links
*		 - eliminated nvar's, modified uvar's
*		 - initialize isp
* Ultrix/RISC port - 2/2/91 - Ultrix compiler doesnt permit variable to be
*               defined multiple times by DATA statements
*               Unix version requires implicit DO loops to be spelled out to
*               avoid duplicate initialization
* V230:  1/30/92 - Sun port: made initialization limits on mr_protected explicit
* V312: 5/94 - array "memory" no longer a COMMON variable
* Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'	! .cmn_text has an external
!	external xgt_grid_data	! xtm_grid --> xgt  (DEC f77 3.2 not allowed)
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'	! .CMN has an EXTERNAL
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

	INTEGER	    max_mrm1
	PARAMETER ( max_mrm1 = max_mrs - 1 )

	INTEGER	i

* dummy stored memory variable
	DATA mr_protected	( dummy_mr ) / mr_perm_protected /
	DATA mr_blk1		( dummy_mr ) / 1			/
	DATA mr_nblks		( dummy_mr ) / 0			/
	DATA mr_data_set	( dummy_mr ) / unspecified_int4	/
	DATA mr_category	( dummy_mr ) / cat_dummy_var	/
	DATA mr_variable	( dummy_mr ) / unspecified_int4	/
	DATA mr_grid		( dummy_mr ) / unspecified_int4	/
	DATA mr_bad_data	( dummy_mr ) / bad_val4		/
	DATA (mr_lo_ww(i,dummy_mr),i=1,4)/ 4*unspecified_val8	/
	DATA (mr_hi_ww(i,dummy_mr),i=1,4)/ 4*unspecified_val8	/
	DATA (mr_trans(i,dummy_mr),i=1,4)/ 4*trans_no_transform /
	DATA (mr_lo_ss(dummy_mr,i),i=1,4)/ 4*1 / ! usable dimension
	DATA (mr_hi_ss(dummy_mr,i),i=1,4)/ 4*1 / ! usable dimension

* interp stack pointer at stack base
	DATA isp / 1 /

* bulk storage initialization
	DATA (mblk_flink(i),i=1,pmax_mem_blks)
     .			/ pmax_mem_blks*unspecified_int4 /
	DATA (mblk_blink(i),i=1,pmax_mem_blks)
     .			/ pmax_mem_blks*unspecified_int4 /
	DATA (mblk_size (i),i=1,pmax_mem_blks)
     .			/ pmax_mem_blks*unspecified_int4 /

* memory variable table and delete priority chain initialization
* (1/92 - made initialization limits explicit for Sun port)
	DATA (mr_protected(i),i=1,max_mrm1)
     .	      / max_mrm1*mr_deleted /	! top one is mr_dummy

	DATA mr_del_flink(0)/ 0 /,
     .	     mr_del_blink(0)/ 0 /,
     .	     (mr_del_flink(i),i=1,max_mrs)/ max_mrs*unspecified_int4 /,
     .	     (mr_del_blink(i),i=1,max_mrs)/ max_mrs*unspecified_int4 /

* user-defined variables
	DATA uvar_num_items	/ max_uvar*uvar_deleted /
	DATA num_uvars_in_cmnd	/ cmnd_uvars_not_given	/
	END
