	LOGICAL FUNCTION ITSA_MODULO_CURVI_AXIS( xfield, yfield, mr_xpos, 
     .                                           mr_ypos, idim, 
     .                                           dim1, dim2)

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
*       determine whether a particular curvilinear context grid axis is a modulo one

*       programmer - jonathan callahan
*       NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*       V420: 04/97
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V687 *acm* 3/14  Ticket 2151: The coordinate variables may be in other dimensions
*                  than XY, e.g. XZ or YZ for a layerz-coordinate vertical axis.
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'tmap_errors.parm'
	include 'xtm_grid.cmn_text'
	include 'xcurvilinear.cmn'

*       calling argument declarations:
	LOGICAL ITSA_MODULO_CURVI_AXIS_SUB
	REAL    xfield(*), yfield(*)
	INTEGER	mr_xpos, mr_ypos, idim, dim1, dim2

*	Internal declarations
	INTEGER slox_dim1, shix_dim1, slox_dim2, shix_dim2, 
     .          sloy_dim1, shiy_dim1, sloy_dim2, shiy_dim2

* The typical xlon, ylat case
	IF (dim1 .EQ. 1) THEN
	   slox_dim1 = mr_lo_s1(mr_xpos)
	   shix_dim1 = mr_hi_s1(mr_xpos)
	   sloy_dim1 = mr_lo_s1(mr_ypos)
	   shiy_dim1 = mr_hi_s1(mr_ypos)
	ENDIF

* Coordinate field 1 represents something else
	IF (dim1 .EQ. 2) THEN
	   slox_dim1 = mr_lo_s2(mr_xpos)
	   shix_dim1 = mr_hi_s2(mr_xpos)
	   sloy_dim1 = mr_lo_s2(mr_ypos)
	   shiy_dim1 = mr_hi_s2(mr_ypos)
	ENDIF
	IF (dim1 .EQ. 3) THEN
	   slox_dim1 = mr_lo_s3(mr_xpos)
	   shix_dim1 = mr_hi_s3(mr_xpos)
	   sloy_dim1 = mr_lo_s3(mr_ypos)
	   shiy_dim1 = mr_hi_s3(mr_ypos)
	ENDIF
	IF (dim1 .EQ. 4) THEN
	   slox_dim1 = mr_lo_s4(mr_xpos)
	   shix_dim1 = mr_hi_s4(mr_xpos)
	   sloy_dim1 = mr_lo_s4(mr_ypos)
	   shiy_dim1 = mr_hi_s4(mr_ypos)
	ENDIF
	IF (dim1 .EQ. 5) THEN
	   slox_dim1 = mr_lo_s5(mr_xpos)
	   shix_dim1 = mr_hi_s5(mr_xpos)
	   sloy_dim1 = mr_lo_s5(mr_ypos)
	   shiy_dim1 = mr_hi_s5(mr_ypos)
	ENDIF
	IF (dim1 .EQ. 6) THEN
	   slox_dim1 = mr_lo_s6(mr_xpos)
	   shix_dim1 = mr_hi_s6(mr_xpos)
	   sloy_dim1 = mr_lo_s6(mr_ypos)
	   shiy_dim1 = mr_hi_s6(mr_ypos)
	ENDIF


* The typical xlon, ylat case
	IF (dim2 .EQ. 2) THEN
	   slox_dim2 = mr_lo_s2(mr_xpos)
	   shix_dim2 = mr_hi_s2(mr_xpos)
	   sloy_dim2 = mr_lo_s2(mr_xpos)
	   shiy_dim2 = mr_hi_s2(mr_xpos)
	ENDIF

* Coordinate field 2 represents something else
	IF (dim2 .EQ. 1) THEN
	   slox_dim2 = mr_lo_s1(mr_xpos)
	   shix_dim2 = mr_hi_s1(mr_xpos)
	   sloy_dim2 = mr_lo_s1(mr_xpos)
	   shiy_dim2 = mr_hi_s1(mr_xpos)
	ENDIF
	IF (dim2 .EQ. 3) THEN
	   slox_dim2 = mr_lo_s3(mr_xpos)
	   shix_dim2 = mr_hi_s3(mr_xpos)
	   sloy_dim2 = mr_lo_s3(mr_xpos)
	   shiy_dim2 = mr_hi_s3(mr_xpos)
	ENDIF
	IF (dim2 .EQ. 4) THEN
	   slox_dim2 = mr_lo_s4(mr_xpos)
	   shix_dim2 = mr_hi_s4(mr_xpos)
	   sloy_dim2 = mr_lo_s4(mr_xpos)
	   shiy_dim2 = mr_hi_s4(mr_xpos)
	ENDIF
	IF (dim2 .EQ. 5) THEN
	   slox_dim2 = mr_lo_s5(mr_xpos)
	   shix_dim2 = mr_hi_s5(mr_xpos)
	   sloy_dim2 = mr_lo_s5(mr_xpos)
	   shiy_dim2 = mr_hi_s5(mr_xpos)
	ENDIF
	IF (dim2 .EQ. 6) THEN
	   slox_dim2 = mr_lo_s6(mr_xpos)
	   shix_dim2 = mr_hi_s6(mr_xpos)
	   sloy_dim2 = mr_lo_s6(mr_xpos)
	   shiy_dim2 = mr_hi_s6(mr_xpos)
	ENDIF

	ITSA_MODULO_CURVI_AXIS = ITSA_MODULO_CURVI_AXIS_SUB( xfield, yfield,
     .          slox_dim1, shix_dim1, slox_dim2, shix_dim2, 
     .          sloy_dim1, shiy_dim1, sloy_dim2, shiy_dim2,
     5    idim )

	RETURN
	END
	
