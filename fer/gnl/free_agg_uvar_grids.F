	SUBROUTINE FREE_AGG_UVAR_GRIDS( dset, status )

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
* if LET/D definitions (uvars) are among the variables in an E or F aggregation
* then the use counts of the grids of those uvars were bumped in during the 
* initialization of the agg (in DEFINE_EF_AGGREGARTION_GRID).
* Here we undo that -- reducing the use count of those grids

* V698 *sh* 2/16 

* calling arguments
*  dset      -  (input) dataset number to be closed
*  status    - (output) return status

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xdset_parms.cmn'
	include 'xprog_state.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xrisc.cmn'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	INTEGER	dset, status

* local variable declarations:
	INTEGER nmemb, imemb, ivar,
     .          memb_cat, memb_dset, memb_grid, iline, memb_var

	IF (ds_type(dset).EQ.'ENS' .OR. ds_type(dset).EQ.'FCT') THEN
	   CALL CD_GET_AGG_DSET_INFO (dset, nmemb, status)
	   IF (status .NE. ferr_ok) GOTO 5100
!   ... note:  this is a very inefficient loop. Better to use the LISTs
!              that knows which variables belong to each dataset 
	   DO ivar = 1,maxvars
	      IF (ds_var_setnum(ivar) .NE. dset) CYCLE
	      DO imemb = 1, nmemb
	         CALL CD_GET_AGG_VAR_INFO (dset, ds_var_code(ivar), imemb, 
     .           memb_cat, memb_dset, memb_grid, iline, memb_var, status)
	         IF (status .NE. ferr_ok) GOTO 5100
	         IF (memb_cat .EQ. cat_user_var) CALL TM_DEALLO_DYN_GRID
     .								(memb_grid)
	      ENDDO
	   ENDDO
	ENDIF

        RETURN

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_internal, status, 'free_uvar_agg_grids', *5000 )

	END
