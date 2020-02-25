	LOGICAL FUNCTION TM_ITSA_DSG ( grid )

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
* determine if this grid represents a DSG data set

* V720 *sh* 4/17
* V745 *acm* 2/19 check that there's a grid defined before using it
* V76  1/20 *acm* working with Point-type dsg data 

      include 'netcdf.inc'
      include 'tmap_errors.parm'
      include 'tmap_dims.parm'
      include 'implicit.parm'   ! pline_not_dsg
#include "gt_lib.parm"
      include 'xtm_grid.cmn_text'
      include 'xdset_info.cmn_text'

* argument declarations
	INTEGER grid

* internal variable declarations
	INTEGER line, dset, orientation, idim

	TM_ITSA_DSG = .FALSE.
	IF (grid .EQ. unspecified_int4) RETURN

* to be a DSG grid there must be an E axis
	line = grid_line(e_dim,grid)
	IF (line .EQ. mpsnorm) THEN
	   TM_ITSA_DSG = .FALSE.
	   RETURN
	ENDIF

* line_dsg_dset is either a DSG dset number or a "not DSG" flag
	dset = line_dsg_dset(line)
	IF (dset .EQ. pline_not_dsg) THEN
	   TM_ITSA_DSG = .FALSE.
	   RETURN
	ENDIF

* now check the orientation axis
	orientation = dsg_orientation(dset)
	line = grid_line(orientation,grid)
	IF (line .EQ. mpsnorm) THEN
	   TM_ITSA_DSG = .TRUE.    ! it's an instance grid
	ELSEIF(line_dsg_dset(line) .EQ. pline_not_dsg) THEN
	   TM_ITSA_DSG = .FALSE.   ! it's a DSG regridded to a classic grid
	ELSE
	   TM_ITSA_DSG = .TRUE.    ! it's an observations grid
	ENDIF

	IF (orientation .EQ. e_dim) THEN
	   DO idim = 1, t_dim 
	      IF (grid_line(idim,grid) .NE. mpsnorm) TM_ITSA_DSG = .FALSE.
	   ENDDO
	ENDIF


	RETURN

        END
