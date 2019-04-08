	BLOCK DATA XVARIABLES_DATA

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
*       2/12 *kms* Add E and F dimensions (set to nferdims in tmap_dims.parm)
* V685+: 3/14 *sh* - initialize max_usable_intrp
* V695+: 5/15 *sh* - initialize uvar_c_cache_ptr
* V702:  2/17 *sh* - remove old block memory management variables
* V745  4/19 *acm* ticket 1916, special handling getting context for RETURN= outputs
*                  new dynamic-memory line storage for keeping a list of uvars

        IMPLICIT NONE
#include "tmap_dset.parm"
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'	! .cmn_text has an external
!	external xgt_grid_data	! xtm_grid --> xgt  (DEC f77 3.2 not allowed)
	include 'ferret.parm'
	include 'xvariables.cmn'

	INTEGER	    max_mrm1
	PARAMETER ( max_mrm1 = max_mrs - 1 )

	INTEGER	i

* dummy stored memory variable
	DATA mr_protected	( dummy_mr ) / mr_perm_protected /
	DATA mr_size		( dummy_mr ) / 0		/
	DATA mr_data_set	( dummy_mr ) / unspecified_int4	/
	DATA mr_category	( dummy_mr ) / cat_dummy_var	/
	DATA mr_variable	( dummy_mr ) / unspecified_int4	/
	DATA mr_grid		( dummy_mr ) / unspecified_int4	/
	DATA mr_bad_data	( dummy_mr ) / bad_val4		/
	DATA (mr_lo_ww(i,dummy_mr),i=1,nferdims)/ nferdims*unspecified_val8 /
	DATA (mr_hi_ww(i,dummy_mr),i=1,nferdims)/ nferdims*unspecified_val8 /
	DATA (mr_trans(i,dummy_mr),i=1,nferdims)/ nferdims*trans_no_transform /
	DATA (mr_lo_ss(dummy_mr,i),i=1,nferdims)/ nferdims*1 / ! usable dimension
	DATA (mr_hi_ss(dummy_mr,i),i=1,nferdims)/ nferdims*1 / ! usable dimension

* interp stack pointer at stack base
	DATA isp / 1 /

* interp stack pointer movable top
	DATA max_usable_intrp / max_intrp /

* memory variable table and delete priority chain initialization
* (1/92 - made initialization limits explicit for Sun port)
	DATA (mr_protected(i),i=1,max_mrm1)
     .	      / max_mrm1*mr_deleted /	! top one is mr_dummy

	DATA mr_del_flink(0)/ 0 /,
     .	     mr_del_blink(0)/ 0 /,
     .	     (mr_del_flink(i),i=1,max_mrs)/ max_mrs*unspecified_int4 /,
     .	     (mr_del_blink(i),i=1,max_mrs)/ max_mrs*unspecified_int4 /

* user-defined variables
        DATA uvar_name_code_head / 0 /
        DATA uvar_name_code_padding / 0 /
        DATA uvar_name_code      /max_uvar*' '/
        DATA uvar_num_items_head / 0 /
        DATA uvar_num_items_padding / 0 /
	DATA uvar_num_items	/ max_uvar*uvar_deleted /
	DATA uvar_c_cache_ptr	/ max_uvar*0.0 /
	DATA num_uvars_in_cmnd	/ cmnd_uvars_not_given	/

	
* Remote user-defined variables: LET/REMOTE
	DATA uvar_remote	 /max_uvar*.FALSE./
	DATA rvar_on_server	 /max_rvar*.FALSE./
	DATA rvar_uvar		 /max_rvar*0/
	DATA rvar_dset		 /max_rvar*0/
	DATA rvar_varid		 /max_rvar*0/

	END
