	BLOCK DATA XVARIABLES_DATA

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
#	include "tmap_dset.parm"
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'	! .cmn_text has an external
!	external xgt_grid_data	! xtm_grid --> xgt  (DEC f77 3.2 not allowed)
	include 'ferret.parm'
	include 'xvariables.cmn'
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
