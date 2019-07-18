	SUBROUTINE REGRID_FROM_DSG_LIMS( dst_cx, src_cx, status )

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
* given a context describing data to be regridded FROM a destination DSG grid
* to a uniform axis along the feature orientation, set up the DSG source
* data context
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
	INTEGER	KNOWN_GRID,
     .		idim, src_grid

* initialize

* install the native grod for the source data
	src_grid = KNOWN_GRID( cx_data_set(src_cx),
     .			       cx_category(src_cx),
     .			       cx_variable(src_cx))
	cx_grid(src_cx) = src_grid
	cx_unstand_grid( src_cx ) = .FALSE.

* set it up as an ordinary DSG context -- preserve E and F from user settings
	DO idim = 1, 4
	   cx_lo_ww(idim,src_cx) = unspecified_val8
	   cx_hi_ww(idim,src_cx) = unspecified_val8
	ENDDO
	DO idim = 1, nferdims
	   CALL FLESH_OUT_AXIS( idim, src_cx, status )
	   IF (status .NE. ferr_ok) GOTO 5000
	ENDDO

* success
	RETURN

* error exit
 5000	RETURN

	END
