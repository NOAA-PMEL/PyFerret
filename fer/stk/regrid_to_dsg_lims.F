	SUBROUTINE REGRID_TO_DSG_LIMS( dst_cx, src_cx, status )

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
* given a context describing data to be regridded to a destination DSG grid
* from a traditional grid, set up the range of source data needed 
* When called the source context should be a copy of the destination context 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* V720 *sh* 5/17

	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'
	include 'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	src_cx, dst_cx, status

* internal variable declarations:
	INTEGER	KNOWN_GRID, CGRID_AXIS, TM_LENSTR1, TM_DSG_DSET_FROM_GRID,
     .		idim, src_var, src_cat, slen,
     .		src_grid, dsg_grid, dsg_dset, xlate_grid
	REAL*8	TDEST_CONVERT
	CHARACTER*32 VAR_CODE, vname

* initialize
	src_cat = cx_category(src_cx)
	src_var = cx_variable(src_cx)
	dsg_grid = cx_grid(dst_cx)
	dsg_dset = TM_DSG_DSET_FROM_GRID(dsg_grid)
	xlate_grid = dsg_xlate_grid(dsg_dset)

* install the native grod for the source data
	src_grid = KNOWN_GRID( cx_data_set(src_cx),
     .					src_cat,
     .					src_var	)
	cx_grid(src_cx) = src_grid
	cx_unstand_grid( src_cx ) = .FALSE.

* Get the limits of gridded source data needed.
* Every axis of the source data must be specified in some way,
* either through the WW limits of the destination DSG context
* or through a LET definition on the source data.
* For example, if the DSG has only XYT axes, and the source grid is XYZT
* then source_var[G=my_dsg] is an error.  The error can be resolved by the user
* by issuing LET var_atZ0 = source_var[Z=0] and regridding var_atZ0[G=my_dsg]
	DO idim = 1, 4
	   IF (cx_regrid_trans(idim,dst_cx) .EQ. unspecified_int4) THEN
	      IF (CGRID_AXIS(idim,src_cx) .NE. mnormal) THEN
	         IF (src_cat .EQ. cat_file_var
     .		  .OR.  (src_cat .EQ. cat_user_var
     .		   .AND. uvar_given(idim,src_var) .GE. uvlim_needed) )
     .								GOTO 5100
	      ENDIF
	      cx_by_ss(idim,src_cx) = .TRUE.
	      cx_lo_ss(src_cx,idim) = unspecified_int4
	      cx_hi_ss(src_cx,idim) = unspecified_int4
	   ELSE
	      cx_regrid_trans(idim,src_cx) = unspecified_int4
	      cx_by_ss(idim,src_cx) = .FALSE.
	      IF (idim .EQ. t_dim) THEN
* ... need to convert from dest time encoding to source time encoding
! DSGTBD The equivalent time conversion code in RQST_REGRID is very much longer
!  It has a lot of logic pertaining to calendars. I see that tdest_world_init.F
!  has code that seems to support calendars.  Am I doing something wrong here?
	         CALL TDEST_WORLD_INIT (xlate_grid, src_grid, t_dim, status )
	         IF (status .NE. ferr_ok) GOTO 5000
	         cx_lo_ww(idim,src_cx) = TDEST_CONVERT(cx_lo_ww(idim,dst_cx))
	         cx_hi_ww(idim,src_cx) = TDEST_CONVERT(cx_hi_ww(idim,dst_cx))
	      ELSE
	         cx_lo_ww(idim,src_cx) = cx_lo_ww(idim,dst_cx)
	         cx_hi_ww(idim,src_cx) = cx_hi_ww(idim,dst_cx)
	      ENDIF
	      cx_by_ss(idim,src_cx) = .FALSE.
	      CALL CONFINE_AXIS( idim, src_cx, status )
	      IF (status .NE. ferr_ok) GOTO 5000
	   ENDIF
	   CALL FLESH_OUT_AXIS( idim, src_cx, status )
* ... the ss limits are now correct but the ww limits are the DSG dset limts
*     Call FLESH_OUT_AXIS again to shift WW limits to align with grid box edges
! 5/17 *sh* I have never tried this trick approach.  If there are problems
!     on modulo axes of whatever, then we need to go step by step here using
!     ISUBSCRIPT's rounding capabilities to make sure we get both correct
!     index values and world coordinate limits
	   cx_by_ss(idim,src_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS( idim, src_cx, status )
	   IF (status .NE. ferr_ok) GOTO 5000
	ENDDO

* success
	RETURN

* error exit
 5000	RETURN
 5100   vname = VAR_CODE(cx_category(src_cx),cx_variable(src_cx))
	slen = TM_LENSTR1(vname) 
	CALL ERRMSG( ferr_regrid, status,
     .          ww_dim_name(idim)//' axis of '//vname(:slen)
     .		//' requires a point location for regridding', *5000)

	END
