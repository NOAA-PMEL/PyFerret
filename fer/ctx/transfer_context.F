	SUBROUTINE TRANSFER_CONTEXT ( source, dest )

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
* transfer space/time/data_set/variable context from one context buffer to
* another

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/27/86
* revision 0.1 - 8/18/87 - added cx_trans_arg
* revision 1.0 - 9/23/87 - special logic for pre-defined contexts (dest<1)
* revision 1.1 - 3/4/88  - corrected .LT. --> .LE. on cx value
* revision 1.2 - 1/4/89  - added regridding transforms
* V200:  5/31/89 - 4D symmetrical (using TRANSFER_AXIS)
*		 - added cx_ovsiz, eliminated %%_relative
* V420: 9/95	- Support for implicit grids (cx_has_impl_grid)
* V530:  8/00 *sh* - transfer data type, too
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V685 3/13 *sh* support for aux var regridding: var[gz(depth)=zax]
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'

	INTEGER	source, dest, idim, i

* all contexts contain axis region information
	DO idim = 1, nferdims
	   CALL TRANSFER_AXIS( idim, source, dest )
	ENDDO

* that's all that pre-defined contexts have
	IF ( source .LE. 0 .OR. dest .LE. 0 ) RETURN

	DO idim = 1, nferdims
	   cx_delta   ( idim, dest ) = cx_delta   ( idim, source )
	   cx_regrid_trans(idim,dest)=cx_regrid_trans(idim,source)
	ENDDO

	cx_data_set( dest ) = cx_data_set( source )
	cx_dset_gvn( dest ) = cx_dset_gvn( source )
	cx_category( dest ) = cx_category( source )
	cx_variable( dest ) = cx_variable( source )
	cx_grid    ( dest ) = cx_grid    ( source )
	cx_type    ( dest ) = cx_type    ( source )
	cx_ovsiz   ( dest ) = cx_ovsiz   ( source )
	cx_bad_data( dest ) = cx_bad_data( source )

	cx_unstand_grid ( dest ) = cx_unstand_grid ( source )
	cx_impl_grid    ( dest ) = cx_impl_grid    ( source )

* do not inherit cx_has_impl_grid.  It is a flag to say which cx
* actually  **allocated** the grid
*	cx_has_impl_grid( dest ) = .FALSE.

* transfer aux var info
	cx_naux( dest ) = cx_naux( source )
	DO i = 1, nferdims
	   cx_aux_arg_start(i, dest) = cx_aux_arg_start(i, source)
	   cx_aux_arg_end  (i, dest) = cx_aux_arg_end  (i, source)
	   cx_aux_stat     (i, dest) = cx_aux_stat     (i, source)
	ENDDO


	RETURN
	END
