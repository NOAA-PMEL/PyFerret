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

#ifdef unix
#include "tmap_dset.parm"		
	include 'tmap_dims.parm'		! parameter definitions
	include 'xdset_info.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'		! parameter definitions
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! parameter definitions
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
#endif

* local parameters
	INTEGER		max6axes, max4vars, maxdsets_aux, maxdsets_parm
	PARAMETER	(max6axes = 6*maxdsets)
	PARAMETER	(max4vars = 4*maxvars)
	PARAMETER	(maxdsets_aux = maxdsets*num_aux)
	PARAMETER	(maxdsets_parm = maxdsets*num_parm)

* Initialize variables associated with data set
	DATA	ds_type			/maxdsets*char_init04/
	DATA	ds_format		/maxdsets*char_init04/
	DATA	ds_source_class		/maxdsets*char_init16/
	DATA	ds_source		/maxdsets*char_init16/
	DATA	ds_subsource		/maxdsets*char_init16/
	DATA	ds_prog_rev		/maxdsets*char_init04/
********************************************************************************
	DATA	ds_expnum		/maxdsets*char_init04/
	DATA	ds_modnum		/maxdsets*char_init04/
	DATA	ds_title		/maxdsets*char_init80/
	DATA	ds_mod_title		/maxdsets*char_init80/
	DATA	ds_forcing_set		/maxdsets*char_init16/
	DATA	ds_t0time		/maxdsets*char_init20/
	DATA	ds_time_unit		/maxdsets*real4_init/
	DATA	ds_time_modulo		/maxdsets*.FALSE./
	DATA	ds_ntegrate_tstep	/maxdsets*real4_init/
	DATA	ds_ntuple		/maxdsets*int4_init/
	DATA	ds_time_run		/maxdsets*char_init20/
	DATA	ds_aux_modnum		/maxdsets_aux*char_init04/
	DATA	ds_parm_name		/maxdsets_parm*char_init20/
	DATA	ds_parm_text		/maxdsets_parm*char_init80/
********************************************************************************
	DATA	ds_alert_on_open	/maxdsets*.FALSE./
	DATA	ds_alert_on_output	/maxdsets*.FALSE./
	DATA	ds_message		/maxdsets*char_init80/
********************************************************************************
	DATA	ds_des_name		/maxdsets*char_init128/
	DATA	ds_name			/maxdsets*char_init40/
	DATA	ds_1step		/maxdsets*real8_init/
	DATA	ds_lastep		/maxdsets*real8_init/
	DATA	ds_delstep		/maxdsets*real8_init/
	DATA	ds_regsteps		/maxdsets*.TRUE./
	DATA	ds_basic_axes		/max6axes*int4_init/
	DATA	ds_time_axis		/maxdsets*int4_init/

* Next clear variables associated with variable in that data set.
	DATA	ds_var_code		/maxvars*char_init08/
	DATA	ds_var_title		/maxvars*char_init40/
	DATA	ds_var_titl_mod		/maxvars*char_init40/
	DATA	ds_var_units		/maxvars*char_init24/
	DATA	ds_grid_number		/maxvars*int4_init/
	DATA	ds_missing_flag		/maxvars*real4_init/
	DATA	ds_bad_flag		/maxvars*real4_init/
	DATA	ds_ordering		/max4vars*int4_init/
	DATA	ds_grid_start		/max4vars*int4_init/
	DATA	ds_grid_end		/max4vars*int4_init/
	DATA	ds_precision		/maxvars*char_init01/
	DATA	ds_aux_set_num		/maxvars*int4_init/
********************************************************************************
	DATA	ds_ndataperrec		/maxvars*int4_init/
	DATA	ds_nrecpervar		/maxvars*int4_init/
	DATA	ds_npospervar		/maxvars*int4_init/
	DATA	ds_nrecb4var		/maxvars*int4_init/
	DATA	ds_var_setnum		/maxvars*set_not_open/
********************************************************************************
	DATA	min_ds_tstart		/int4_init/
	DATA	max_ds_tend		/int4_init/
	DATA	ds_tdelta		/int4_init/

	END
