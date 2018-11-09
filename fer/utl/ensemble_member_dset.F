	INTEGER FUNCTION ENSEMBLE_MEMBER_DSET ( agg_dset, imemb )

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
* return the dset number of ensemble member imemb
* note:  as currently implemented an aggregated dataset knows who its member
*        variable are, and each member variable knows who its member datasets
*        are.  So the only way to learn the dataset number of a member is to
*        access it through a member variable.  This routine follows that path.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*

* calling arguments
*  agg_dset -- dataset number of the aggregation
*  imemb    -- member number 1, 2, 3, ...


* V690 *sh*

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'xdset_info.cmn_text'
	include 'xtm_grid.cmn_text'

* calling argument declarations
	INTEGER agg_dset, imemb	

* local variable declarations:
	INTEGER TM_LENSTR1
	INTEGER ivar, grid, vlen, memb_dset
	INTEGER orig_cat, orig_grid, iline, orig_var, status


* find a variable that belongs to this dataset and has an E axis
! very inefficient loop:  needs to be fixed
	DO ivar = 1, maxvars
	   IF (ds_var_setnum(ivar).EQ.agg_dset) THEN
	      grid = ds_grid_number(ivar)
	      IF (grid_line(e_dim,grid) .NE. mnormal) GOTO 100
	   ENDIF
	ENDDO
* ... no variable found (should never happen)
	ENSEMBLE_MEMBER_DSET = pdset_irrelevant ! may avoid a crash sometime?
	RETURN 

* get the dataset number associated with this variable
 100	vlen = TM_LENSTR1( ds_var_code(ivar) )
	CALL CD_GET_AGG_VAR_INFO (agg_dset, ds_var_code(ivar)(:vlen), imemb, 
     .        orig_cat, memb_dset, orig_grid, iline, orig_var, status)
	ENSEMBLE_MEMBER_DSET = memb_dset
	RETURN
	END

