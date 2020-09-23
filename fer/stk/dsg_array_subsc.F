	SUBROUTINE DSG_ARRAY_SUBSC( mr_list, nmr, orientation )

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
* Alter the subscript ranges set by NON_ARRAY_SUBSC into a 2D form
* dimension 1 - either the E or the orientation axis (whichever is relevant)
* dimension 2 - F
* The orientation dimension is the z-direction for trajectory-profile 
* and timeseries-profile feature types

* Arguably a stylisticly nicer and more generic name for this routine might
* have been CONSOLIDATE_ARRAY_SUBSCRIPTS, as its function is nor DSG-specific.
* This routine merely makes 1D arrays look like 1D arrays rather than 6D
* arrays with a variable position for their only non-normal dimension.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V720 *sh* 5/17
* V74+ *acm* 4/18  fix for computation with an instance-only variable

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include 'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER nmr, mr_list(nmr), orientation

* internal variable declarations:
	INTEGER swap_dim, ss, mr, orient

* now consolidate the E and the orientation axes into the Xdim
* note that the orientation may already be x_dim. make sure to preserve if so

	orient = orientation
	IF (orient.EQ.y_dim .OR. orient.EQ.f_dim) orient = z_dim

	DO  mr  = 1, nmr
	   IF (hi_ss(mr,orient) .NE. unspecified_int4) THEN
* ... a ragged array "observation" variable
	      swap_dim = orient
	   ELSE
	      swap_dim = e_dim
	   ENDIF
	   ss = lo_ss( mr, x_dim )
	   lo_ss( mr, x_dim ) = lo_ss( mr, swap_dim )
	   IF (ss .NE. unspecified_int4) lo_ss( mr, orient ) = ss 

	   ss = hi_ss( mr, x_dim )
	   hi_ss( mr, x_dim ) = hi_ss( mr, swap_dim )
	   IF (ss .NE. unspecified_int4) hi_ss( mr, orient ) = ss 
	ENDDO

	RETURN
	END
