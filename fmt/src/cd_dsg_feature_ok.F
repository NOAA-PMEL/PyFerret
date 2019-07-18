        LOGICAL FUNCTION CD_DSG_FEATURE_OK (dset, dsg_direction)


*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* Return TRUE if the direction is the native direction of the ragged-array
* file, or if it can be treated as the requested direction, e.g. a trajectory
* dataset treated as a timeseries dataset.

* Programmer Ansley Manke SDIG 9/2018

* argument definitions:
*       dset      - dataset number for already opened CDF file
*       dsg_direction      - feature-direction flag, 
*	      trajectory marked as x_dim = 1 
*	      profile	 marked as z_dim = 3
*	      time	 marked as t_dim = 4
*	      point	 not yet handled

* include files
        include 'netcdf.inc'
        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xio.cmn_text'
        external xio_data
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xdyn_linemem.cmn_text'
        include 'xtm_grid.cmn_text'

* argument declarations
        INTEGER  dset, dsg_direction

* local variable declarations:
        INTEGER i, lm, obs_dimlen, ivar

* If the set is already open but with no orientation (from a use/feature=none)
* or if the set is already open with the same direction as being requested, we're done.

	IF  (dsg_orientation(dset) .LE. no_dim) THEN
	   CD_DSG_FEATURE_OK = .TRUE.
	   GOTO 5000
	ENDIF

	IF  (dsg_direction .EQ. dsg_orientation(dset)) THEN
	   CD_DSG_FEATURE_OK = .TRUE.
	   GOTO 5000
	ENDIF

        obs_dimlen = dsg_obs_dimlen(dset)

	IF (dsg_direction .EQ. x_dim) THEN

	   ivar = dsg_coord_var(x_dim,dset)
           lm = dsg_loaded_lm(ivar) 
	   IF (lm_size(lm) .NE. obs_dimlen) GOTO 5100

	   ivar = dsg_coord_var(y_dim,dset)
	   IF (ivar .EQ. int4_init) GOTO 5100
           lm = dsg_loaded_lm(ivar) 
	   IF (lm_size(lm) .NE. obs_dimlen) GOTO 5100

	   ivar = dsg_coord_var(t_dim,dset)
           lm = dsg_loaded_lm(ivar) 
	   IF (lm_size(lm) .NE. obs_dimlen) GOTO 5100

        ElSE IF (dsg_direction .EQ. z_dim) THEN

	   ivar = dsg_coord_var(z_dim,dset)
	   IF (ivar .EQ. int4_init) GOTO 5200
           lm = dsg_loaded_lm(ivar) 
	   IF (lm_size(lm) .NE. obs_dimlen) GOTO 5200

	ELSE IF (dsg_direction .EQ. t_dim) THEN

	   ivar = dsg_coord_var(t_dim,dset)
	   IF (ivar .EQ. int4_init) GOTO 5300
           lm = dsg_loaded_lm(ivar) 
	   IF (lm_size(lm) .NE. obs_dimlen) GOTO 5300

	ELSE
	   ! Future: point types, others?
	ENDIF
	
	CD_DSG_FEATURE_OK = .TRUE.


 5000   RETURN
 5500   CD_DSG_FEATURE_OK = .FALSE.
	RETURN

 5100	CONTINUE
        CALL TM_NOTE('Coordinates inconsistent with FeatureType. '//
     .    'Trajectory data must have observations along XYT', lunit_errors )
	GOTO 5500
 5200	CONTINUE
        CALL TM_NOTE('Coordinates inconsistent with FeatureType. '//
     .    ' Profile data must have observations along Z', lunit_errors )
        GOTO 5500

 5300	CONTINUE
        CALL TM_NOTE('Coordinates inconsistent with FeatureType. '//
     .    'Timeseries data must have observations along T', lunit_errors )
	GOTO 5500

	END