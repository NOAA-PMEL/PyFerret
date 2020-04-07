	LOGICAL FUNCTION ITSA_REGRID_FROM_DSG ( cx )

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
* determine if this context represents a regrid operation *from* a DSG variable

* V720 *sh* 5/17
* V76  1/20 *acm* working with Point-type dsg data

      include 'tmap_dims.parm'
      include 'implicit.parm'
      include 'ferret.parm'
      include 'xcontext.cmn'
      include 'xtm_grid.cmn_text'
      include 'xdset_info.cmn_text'
      include 'xfr_grid.cmn'

* argument declarations
	INTEGER cx

* internal variable declarations
	INTEGER grid, dset, line, orient

* initialize
	grid = cx_grid(cx)

* is it a regrid at all?
	IF (.NOT.cx_unstand_grid(cx)) THEN
	   ITSA_REGRID_FROM_DSG = .FALSE.
	   RETURN
	ENDIF

* the E axis tells us if the starting point is DSG
	line = grid_line(e_dim,grid)
	IF (line .EQ. mnormal) THEN
	   ITSA_REGRID_FROM_DSG = .FALSE.
	   RETURN
	ENDIF
	dset = line_dsg_dset(line)
	IF ( dset .EQ. pline_not_dsg) THEN
	   ITSA_REGRID_FROM_DSG = .FALSE.
	   RETURN
	ENDIF

* Yes - E tells us that this started as a DSG
* The existence of a feature-oriented axis that is non-DSG indicates a
* regrid from a DSG
* (see comments in dsg_whats_it.F for reasoning behind test for abstract axis)

	orient = dsg_orientation(dset)
	line = grid_line(orient,grid)
	IF (line .EQ. mnormal  .OR. 
     .      line .EQ. grid_line(1,mgrid_xabstract)) THEN
	   ITSA_REGRID_FROM_DSG = .FALSE.
	ELSE
	   ITSA_REGRID_FROM_DSG = line_dsg_dset(line) .EQ. pline_not_dsg
	ENDIF

* We know it's a regrid - the source is a dsg point-type dataset, so the
* grid is fine. (once we implement these regrids!)
	IF (orient .EQ. e_dim) ITSA_REGRID_FROM_DSG = .TRUE.  

        RETURN
        END
