	BLOCK DATA XDSET_INFO_DATA

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
* various pre-defined variables for COMMON/XDSET_INFO
* programmer - mark verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 07/29/86
* revision 0.10 - 05/07/87 - initializes new integration_tstep variable
* revision 0.11 - 07/07/87 - removed revision and increased size of type/format
* revision 0.20 - 08/30/87 - changed to reflect new layout of common
* revision 0.21 - 10/30/87 - eliminated ds_make_dscrp_rev
* revision 0.30 - 07/27/88 - added variables from TS descriptor type
* revision 0.31 - 08/10/88 - made number of aux_parms a parameter, aux_parm now
*			     2 vars ds_aux_name, ds_aux_text
* revision 0.40 - 10/25/88 - Added modulo variables
* revision 0.41 - 11/21/88 - variable initialization changed to char_initXX
* revision 0.42 - 02/08/89 - removed ds_var_modline
* revision 0.43 - 02/15/89 - removed ds_open
* Unix/RISC port - 4/25/91 *sh*: increased ds_name and ds_des_name size
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
* v500 - *kob* 3/99 -  initialize ds_var_code, ds_var_units, ds_var_title
*                      and ds_var_titl_mod w/ char_inits of proper new value
* v500 - *kob* 4/99 - initialize ds_des_name with char_init256	
* v530 - *acm* 1/01 - add ds_cal_name
* v540 *acm* 10/01 increase length of ds_name
* v540 *acm* 10/01  remove VMS include statements
* v552 *acm*  4/03 Initialize ds_var_off, ds_var_scale, ds_var_scaleit
* v552 *acm*  5/03 change ds_var_code length to 128
* v604 *acm* 7/07 increase length of ds_name, ds_des_name
* v695+ *sh* 5/15 initialize new ds_var_c_cache_ptr
* v698 *sh* 12/15 declare larger dset arrays: maxdsets ==> pdsetalloc
* V701  7/16 *kms* Initialize ds_name and ds_des_name to pdsetalloc*char_init2048 
*                  instead of pdsetalloc*char_init1024
* V720   4/17 *sh* add DSG variables 

#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_dims.parm'		! parameter definitions
	include 'xdset_info.cmn_text'

* local parameters
	INTEGER		max6axes, maxnvars, maxdsets_aux, maxdsets_parm, maxdsg_coords
	PARAMETER	(max6axes = 6*pdsetalloc)
	PARAMETER	(maxnvars = nferdims*maxvars)
	PARAMETER	(maxdsets_aux = pdsetalloc*num_aux)
	PARAMETER	(maxdsets_parm = pdsetalloc*num_parm)
	PARAMETER	(maxdsg_coords = 4*pdsetalloc)

* for COMMON/XDSG_INFO/
	DATA	dsg_ragged		/pdsetalloc*.FALSE./
	DATA	dsg_feature_type	/pdsetalloc*int4_init/
	DATA	dsg_row_size_var	/pdsetalloc*int4_init/
	DATA	dsg_orientation		/pdsetalloc*int4_init/
	DATA	dsg_obs_dimlen		/pdsetalloc*int4_init/
	DATA	dsg_xlate_grid		/pdsetalloc*int4_init/
	DATA	dsg_msk_lm		/pdsetalloc*unspecified_int4/
	DATA	dsg_ftrsetmsk_lm	/pdsetalloc*unspecified_int4/
	DATA	dsg_coord_var		/maxdsg_coords *int4_init/
	DATA	dsg_obs_axis		/int4_init/
	DATA	dsg_instance_axis	/int4_init/
	DATA	dsg_ifeature_dim	/int4_init/
	DATA	dsg_instnc_ftrset_dim	/int4_init/
	DATA	dsg_obs_dim		/int4_init/
	DATA	dsg_loaded_lm		/maxvars*int4_init/
	DATA	dsg_user_direction	/int4_init/
	DATA	dsg_ftrset_var		/pdsetalloc*int4_init/
	DATA	dsg_index_var		/pdsetalloc*int4_init/
	DATA	dsg_mask_ftrset_lm	/pdsetalloc*int4_init/

* Initialize variables associated with data set
	DATA	ds_type			/pdsetalloc*char_init04/
	DATA	ds_format		/pdsetalloc*char_init04/
	DATA	ds_source_class		/pdsetalloc*char_init16/
	DATA	ds_source		/pdsetalloc*char_init16/
	DATA	ds_subsource		/pdsetalloc*char_init16/
	DATA	ds_prog_rev		/pdsetalloc*char_init04/
********************************************************************************
	DATA	ds_expnum		/pdsetalloc*char_init04/
	DATA	ds_modnum		/pdsetalloc*char_init04/
	DATA	ds_title		/pdsetalloc*char_init80/
	DATA	ds_mod_title		/pdsetalloc*char_init80/
	DATA	ds_forcing_set		/pdsetalloc*char_init16/
	DATA	ds_t0time		/pdsetalloc*char_init20/
	DATA	ds_time_unit		/pdsetalloc*real4_init/
	DATA	ds_time_modulo		/pdsetalloc*.FALSE./
	DATA	ds_ntegrate_tstep	/pdsetalloc*real4_init/
	DATA	ds_ntuple		/pdsetalloc*int4_init/
	DATA	ds_time_run		/pdsetalloc*char_init20/
	DATA	ds_aux_modnum		/maxdsets_aux*char_init04/
	DATA	ds_parm_name		/maxdsets_parm*char_init20/
	DATA	ds_parm_text		/maxdsets_parm*char_init80/
	DATA	ds_cal_name		/pdsetalloc*'GREGORIAN'/
********************************************************************************
	DATA	ds_alert_on_open	/pdsetalloc*.FALSE./
	DATA	ds_alert_on_output	/pdsetalloc*.FALSE./
	DATA	ds_message		/pdsetalloc*char_init80/
********************************************************************************
	DATA	ds_des_name		/pdsetalloc*char_init2048/
	DATA	ds_name			/pdsetalloc*char_init2048/
	DATA	ds_1step		/pdsetalloc*real8_init/
	DATA	ds_lastep		/pdsetalloc*real8_init/
	DATA	ds_delstep		/pdsetalloc*real8_init/
	DATA	ds_regsteps		/pdsetalloc*.TRUE./
	DATA	ds_accepts_remote	/pdsetalloc*.FALSE./
	DATA	ds_hide			/pdsetalloc*.FALSE./
	DATA	ds_basic_axes		/max6axes*int4_init/
	DATA	ds_time_axis		/pdsetalloc*int4_init/

* Next clear variables associated with variable in that data set.
        DATA    ds_var_code_head        / 0 /
        DATA    ds_var_code_padding     / 0 /
	DATA	ds_var_code		/maxvars*char_init128/
	DATA	ds_var_title		/maxvars*char_init128/
	DATA	ds_var_titl_mod		/maxvars*char_init128/
	DATA	ds_var_units		/maxvars*char_init64/
	DATA	ds_grid_number		/maxvars*int4_init/
	DATA	ds_missing_flag		/maxvars*real4_init/
	DATA	ds_bad_flag		/maxvars*real4_init/
	DATA	ds_ordering		/maxnvars*int4_init/
	DATA	ds_grid_start		/maxnvars*int4_init/
	DATA	ds_grid_end		/maxnvars*int4_init/
	DATA	ds_precision		/maxvars*char_init01/
	DATA	ds_aux_set_num		/maxvars*int4_init/
********************************************************************************
	DATA	ds_ndataperrec		/maxvars*int4_init/
	DATA	ds_nrecpervar		/maxvar2*int4_init/
	DATA	ds_npospervar		/maxvars*int4_init/
	DATA	ds_nrecb4var		/maxvar2*int4_init/
	DATA	ds_var_setnum		/maxvars*set_not_open/
********************************************************************************
	DATA	min_ds_tstart		/int4_init/
	DATA	max_ds_tend		/int4_init/
	DATA	ds_tdelta		/int4_init/
	DATA	ds_var_off		/maxvars*0.0/
	DATA	ds_var_scale		/maxvars*1.0/
	DATA	ds_var_scaleit		/maxvars*.FALSE./
	DATA	ds_var_c_cache_ptr	/maxvars*0.0/

	END
