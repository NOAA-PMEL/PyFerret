	SUBROUTINE FINALIZE

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
* initialize run-time variables and I/O in preparation to begin execution

#ifdef unix
        include 'tmap_errors.parm'  ! error codes
	include 'tmap_dims.parm'
	include 'xio.cmn_text'
        include 'xdset_info.cmn_text'
	external xio_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'gfdl_vms.parm'
	include	'errmsg.parm'
	include	'gfdl.parm'			! from phil/sieg
	include	'xonedim.cmn'		! from phil.sieg
	include 'xprog_state.cmn'
        include 'xalgebra.cmn'
	include 'xvariables.cmn'
	include 'xtoday.cmn'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'  ! error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:GFDL.PARM'			! from Phil/Sieg
	INCLUDE	'FERRET_CMN:XONEDIM.CMN'		! from Phil.Sieg
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
        INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XTODAY.CMN'
	INCLUDE '($SSDEF)'				! LIB_ return status
	INCLUDE '($JPIDEF)'				! for VMS mode code

	INTEGER	SMG$CREATE_VIRTUAL_KEYBOARD, SMG$CREATE_KEY_TABLE,
     .		LIB$SYS_TRNLOG
	INTEGER	vax_code, vms_mode
#endif

        CALL string_array_clear(alg_pvar_head)

        CALL string_array_clear(countervar_name_head)

        CALL string_array_clear(uvar_name_code_head)
 
        CALL string_array_clear(ds_var_code_head)

        CALL deleted_list_clear(uvar_num_items_head)

	RETURN

	END

