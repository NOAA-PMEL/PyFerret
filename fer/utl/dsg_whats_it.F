	INTEGER FUNCTION DSG_WHATS_IT ( grid )

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
* determine the DSG-ness of this grid.  It may be a
*    - non-DSG
*    - DSG instance variable
*    - DSG contiguous ragged array
*    - timeseries- or trajectory- variable in a timseriesProfile or trajectoryProfile set

* V720 *sh* 4/17

      include 'tmap_dims.parm'
      include 'implicit.parm'
      include 'ferret.parm'
      include 'xtm_grid.cmn_text'
      include 'xdset_info.cmn_text'
      include 'xfr_grid.cmn'

* argument declarations
	INTEGER grid

* internal variable declarations
	INTEGER line, dset, orient

* the test of DSG-ness is line_dsg_dset points to dset on the E axis
	line = grid_line(e_dim,grid)
	IF (line .EQ. mnormal) THEN
	   line = grid_line(f_dim,grid)
	   IF (line .EQ. mnormal) THEN
	      DSG_WHATS_IT = pdsg_not_dsg
	   ELSE
               DSG_WHATS_IT = pdsg_fs_dim
	   ENDIF
	ENDIF
	dset = line_dsg_dset(line)
	IF (dset.EQ.pline_not_dsg .OR. dset.EQ.pdsg_not_dsg) THEN
	   DSG_WHATS_IT = pdsg_not_dsg
	   RETURN
	ENDIF

	IF (DSG_WHATS_IT .EQ. pdsg_fs_dim) RETURN

* we are looking at a DSG grid
* the test for contiguous ragged is the existence of a feature axis
* with line_dsg_dset == the dset
* For user-variables that use some grid-changing functions, the
* result is defined as having abstract axes (e.g. STRINDEX(expocode, "PAN")
* has an abstract axis in axes other than the E axis that it gets from expocode.

	orient = dsg_orientation(dset)
	IF (orient.EQ.y_dim .OR. orient.EQ.f_dim) orient = z_dim ! compound feature type
	line = grid_line(orient,grid)
	IF (grid_line(orient,grid) .EQ. mnormal  .OR. 
     .      grid_line(orient,grid) .EQ. grid_line(1,mgrid_xabstract)) THEN
	   DSG_WHATS_IT = pdsg_instance
	ELSEIF (line_dsg_dset(line) .EQ. pline_not_dsg) THEN
	   DSG_WHATS_IT = pdsg_not_dsg
	ELSE
	   DSG_WHATS_IT = pdsg_observation
	ENDIF

	IF (orient .EQ. e_dim) DSG_WHATS_IT = pdsg_instance

        RETURN
        END
