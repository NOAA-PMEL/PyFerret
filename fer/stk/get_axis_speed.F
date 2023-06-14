	SUBROUTINE GET_AXIS_SPEED( dset )

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
* order the axes X,Y,Z and T into array ds_ax_speed with the fastest (axis of
* records) in position 1, the slowest in position 6, and the others
* ordered from fast to slow if possible.
* normal axes will be regarded as fast (but slower than the record axis)

* e.g. the variable TAUX ordered X,Y in a GT set would produce X,Z,Y,T

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/8/89
* V230:  3/2/92  - netCDF support
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 removing use of old GT and TS data formats
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 *acm*  3/13 For Ensemble aggregations, call the ds_type 'ENS'
* V695 *acm*  2/15 For Forecast aggregations, call the ds_type 'FCT'
* V698 *acm* 1/16 Allow for Epic CDF files, ds_type = 'ECDF'
* V7.6 *acm* 1/20 ds_time_axis initialized with int4_init
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
* calling argument declarations:
	INTEGER	dset

* internal variable declarations:
	INTEGER	idim, i, slowdim, igrid, ivar, nexdim, int4_init

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	include 'ferret.parm'
	include 'xax_speed.cmn'
	
	int4_init = -7777

* insert default assumptions
	DO idim = 1, nferdims
	   ds_ax_speed( idim, dset ) = idim
 	ENDDO

* netCDF
	IF( ds_type(dset) .EQ. 'CDF' .OR. 
     .      ds_type(dset) .EQ. 'ECDF' .OR. 
     .      ds_type(dset) .EQ. 'ENS'.OR. 
     .      ds_type(dset) .EQ. 'FCT') THEN
* ... "record" axis is slowest (stored in ds_time_axis)
* ... locate it in a grid to see what role it plays
           IF (ds_time_axis(dset).EQ.unspecified_int4 .OR. ds_time_axis(dset).EQ.int4_init) RETURN
           DO ivar = 1, maxvars
              IF ( ds_var_setnum(ivar) .NE. dset ) CYCLE
              igrid = ds_grid_number(ivar)
              DO idim = 1, nferdims
                 IF (grid_line(idim,igrid).EQ.ds_time_axis(dset)) GOTO 450
	      ENDDO
	   ENDDO
           RETURN

* ... we found the record axis in a grid
 450       slowdim = idim
           ds_ax_speed(6,dset) = slowdim
           nexdim = 1
	   DO idim = 1, nferdims-1
              IF ( idim .EQ. slowdim ) nexdim = nexdim + 1
              ds_ax_speed(idim,dset) = nexdim
              nexdim = nexdim + 1
	   ENDDO

	ENDIF

	RETURN
	END
