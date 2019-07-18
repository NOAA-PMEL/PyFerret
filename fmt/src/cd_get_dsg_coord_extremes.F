        SUBROUTINE CD_GET_DSG_COORD_EXTREMES(dset, ivcd,  ivtm,
     .                             idim, coord_min, coord_max, status)


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
* Return the extreme values of the indicated DSG coordinate variable
* If the variable has an "actual_range" attributes use it.
* Otherwise compute the values from the data.
* Then retain the results in actual_range.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* v720 4/2017 *sh*
* V74+ 4/18 *acm* Dealing with variables that do not have actual_range attribs.
* V74+ 7/18 *acm* Allow for range min = max for vertical, elsewhere this will
*                  be allowed unless its a profile set.
* V74+ 7/18 *acm* Argument idim used for testing which coordinates are required
* v751 *acm*  6/19 Allow for longer strings as attribute values


* argument definitions:
*       dset      - dataset number for already opened CDF file
*       ivcd      - netCDF variable id number         ("cd" for "netCDF")
*       ivtm      - XDSET_INFO COMMON variable number ("tm" for "TMAP")
*	idim	  - direction of this coordinate variable
*       status    - return status

* include files
        include 'netcdf.inc'
        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xio.cmn_text'
        external xio_data
        include 'xdyn_linemem.cmn_text'
        include 'xtm_grid.cmn_text'

* argument declarations
        INTEGER  dset, ivcd, ivtm, idim, status
        REAL     coord_min, coord_max

* local variable declarations:
        LOGICAL NC_GET_ATTRIB, got_it
        INTEGER TM_LENSTR1, i, lm, attlen, attoutflag, vtype, vlen
        CHARACTER    buff*128, vname*128
        REAL    vals(2), coord, bad

* local parameter definitions:
        INTEGER     maxlen
	LOGICAL	    do_warn
	PARAMETER ( maxlen = 128,
     .              do_warn = .TRUE. )

* initialize
        vname = ds_var_code(ivtm)   ! may be >8 chars, but its just for err msg
	vlen = TM_LENSTR1(vname)

* is there an 'actual_range' attribute?

       got_it = NC_GET_ATTRIB( dset, ivcd, 'actual_range',
     .                 do_warn, vname, maxlen, attlen, attoutflag, 
     .                 buff, vals )

* maybe we're done -- either cuz the file contained the actual_range
* attribute, or because this routine has been called earlier to compute it

        IF (got_it) THEN
           IF (vals(2) .LT. vals(1))  GOTO 5001  ! should do more checking than this?
           coord_min = vals(1)
           coord_max = vals(2)
           status = merr_ok
           RETURN
        ENDIF

        lm = dsg_loaded_lm(ivtm)
        bad = ds_bad_flag(ivtm)

! DSGTBD: need to deal with unmatched missing_value and _FillValue
        coord_min = arbitrary_large_val8
        coord_max = arbitrary_small_val8

        DO i = 1, lm_size(lm)
           coord = dsg_linemem(lm)%ptr(i)
           IF (coord .EQ. bad) CYCLE
           IF (coord .LT. coord_min) coord_min = coord
           IF (coord .GT. coord_max) coord_max = coord
        ENDDO

! Invalid coordinates are important if they are in the feature-type direction
! or if its the time coordinate

        IF (coord_min.EQ.arbitrary_large_val8) THEN
	   IF (idim .EQ. t_dim) GOTO 5002
	   IF (idim .EQ. y_dim .AND. dsg_orientation(dset).EQ. x_dim) GOTO 5002
	   IF (idim .EQ. dsg_orientation(dset) ) GOTO 5002
        ELSE

! Store the attribute actual_range

           CALL CD_GET_VAR_TYPE (dset, ivcd, vname, vtype, status)

           attlen = 2
           attoutflag = 1
           vals(1) = coord_min
           vals(2) = coord_max
           CALL CD_PUT_NEW_ATTR (dset, ivcd, 'actual_range', 
     .        vtype, attlen, attoutflag, buff, vals, status) 

           status = merr_ok
        ENDIF

 5000   RETURN

 5001	CONTINUE
        CALL TM_NOTE("Corrupted coordinate in DSG file actual_range attribute for: "
     .                  //vname(:vlen), lunit_errors )
 5002	CONTINUE
        CALL TM_NOTE("Corrupted coordinate in DSG file. No valid data in: "
     .                  //vname(:vlen), lunit_errors )
        status = 0 

	END