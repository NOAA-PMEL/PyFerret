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
* records) in position 1, the slowest in position 4, and the others
* ordered from fast to slow if possible.
* normal axes will be regarded as fast (but slower than the record axis)

* e.g. the variable TAUX ordered X,Y in a GT set would produce X,Z,Y,T

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/8/89
* V230:  3/2/92  - netCDF support

* calling argument declarations:
	INTEGER	dset

* internal variable declarations:
	INTEGER	idim, i, slowdim, igrid, ivar, nexdim

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xax_speed.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT/XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN/FERRET.PARM'
	INCLUDE 'FERRET_CMN:XAX_SPEED.CMN'
#endif

* insert default assumptions
	DO 100 idim = 1,4
 100	ds_ax_speed( idim, dset ) = idim

* GT formatted:
	IF ( ds_type(dset) .EQ. 'GT' ) THEN
* ... base it on first variable in data set
	   DO 200 i = 1, maxvars
 200	   IF ( ds_var_setnum(i) .EQ. dset ) GOTO 210
	   RETURN	! no vars in set
 210	   DO 220 idim = 1, 3
 220	   ds_ax_speed( idim, dset ) = ABS( ds_ordering(idim,i) )

* TS formatted
	ELSEIF( ds_type(dset) .EQ. 'TS' ) THEN
* ... always the same (at the time of this writing)
	   ds_ax_speed(1,dset) = 4
	   DO 300 idim = 1,3
 300	   ds_ax_speed(idim+1,dset) = idim	   

* netCDF
	ELSEIF( ds_type(dset) .EQ. 'CDF' ) THEN
* ... "record" axis is slowest (stored in ds_time_axis)
* ... locate it in a grid to see what role it plays
           IF (ds_time_axis(dset) .EQ. unspecified_int4) RETURN
           DO 410 ivar = 1, maxvars
              IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 410
              igrid = ds_grid_number(ivar)
              DO 400 idim = 1, 4
 400          IF (grid_line(idim,igrid).EQ.ds_time_axis(dset)) GOTO 450
 410       CONTINUE
           RETURN

* ... we found the record axis in a grid
 450       slowdim = idim
           ds_ax_speed(4,dset) = slowdim
           nexdim = 1
	   DO 460 idim = 1,3
              IF ( idim .EQ. slowdim ) nexdim = nexdim + 1
              ds_ax_speed(idim,dset) = nexdim
              nexdim = nexdim + 1
 460       CONTINUE

	ENDIF

	RETURN
	END
