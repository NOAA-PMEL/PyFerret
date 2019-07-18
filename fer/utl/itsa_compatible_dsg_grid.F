	LOGICAL FUNCTION ITSA_COMPATIBLE_DSG_GRID ( g1, g2 )

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
* determine if 2 grids are compatible.  The test is simple: they cannot have
* E axes that are unmatched in terms of DSG-ness

* V720 *sh* 5/17

      include 'tmap_dims.parm'

      include 'ferret.parm'
      include 'xtm_grid.cmn_text'
      include 'xfr_grid.cmn'

* argument declarations
	INTEGER g1, g2

* internal variable declarations
	INTEGER e1, e2, eabstract

* initialize
	e1 = grid_line(e_dim,g1)
	e2 = grid_line(e_dim,g2)
	eabstract = grid_line(e_dim,mgrid_eabstract)

* if either has a normal E axis there is no issue of incompatibility
	IF (e1.EQ.mnormal .OR. e2.EQ.mnormal) THEN
	   ITSA_COMPATIBLE_DSG_GRID = .TRUE.
	   RETURN
	ENDIF
	   
* if either has an abstract E axis there is no issue of incompatibility
* This lets us use functions to define masks in the feature direction.
	IF (e1.EQ.eabstract .OR. e2.EQ.eabstract) THEN
	   ITSA_COMPATIBLE_DSG_GRID = .TRUE.
	   RETURN
	ENDIF

	ITSA_COMPATIBLE_DSG_GRID = e1 .EQ. e2

        RETURN
        END
