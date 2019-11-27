	SUBROUTINE CD_SCAN_AUX_COORDS (dset, cdfid, native_tax, status)


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
* Finds 'coordinates' attributes on variables and sets up linkages to handle
* CF auxiliary coordinates

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 4/15
* V698 *sh* 3/2016 - pass native_tax to CD_1D_LINE_FROM_2D
* V74+ *acm* 12/2018 - issue 1049: Skip over rare case of a string variable
*                      that looks to be a coordinate

* argument definitions:
*       dset        - TMAP data set pointer
*	cdfid       - netCDF open ID (should no longer be needed!!)
*	native_tax  - the official time axis of the dataset
*	status	    - return status

* include files

        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	        include 'xdset_info.cd_equiv'

* argument declarations
	INTEGER	 dset, cdfid, native_tax, status


* local parameter definitions:
	LOGICAL	    do_warn
	INTEGER     str_eq, max_names
	PARAMETER (str_eq = 0)
	PARAMETER (do_warn = .TRUE.)
	PARAMETER (max_names = 8)   ! max num of axis coordinates (variable) names for a variable


* local variable declarations:
	LOGICAL NC_GET_ATTRIB, got_it, coordvar

	INTEGER TM_LENSTR1, STR_CASE_BLIND_COMPARE, 
     .		i, tmvar, ivar, num_names, nvdim, all_outflag,
     .		maxlen, attlen, attoutflag, tmp, cdf_stat, nvatts,
     .		axcalt, axlagf, istat, vartyp, vlen, vdims(8)
	CHARACTER aux_coord_names*500, coord_name(max_names)*128,
     .		  vname*128, name*16
	REAL vals  ! potential numerical attributes

* loop through each variable in this TMAP data set
* if it has a "coordinates" attribute then associate it with that variable
* and flag that variable as an auxiliary coordinate

        DO tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) CYCLE
           ivar = cd_varid(tmvar)

           CALL CD_GET_VAR_INFO (dset, ivar, vname, vartyp, nvdim, vdims,
     .                 nvatts, coordvar, all_outflag, istat)
	   vlen = TM_LENSTR1(vname)
	   IF (vartyp .EQ. ncchar) CYCLE

* THIS BOCK OF CODE WAS STARTED ONLY QUICKLY
* POSTPONED IT UNTIL LATER TO GET JUST THE 2D TIME FMRC FUNCTIONALITY
!* does it point explicitly to auxiliary coordinate  ?
!           maxlen = 240  
!           got_it = NC_GET_ATTRIB( dset, ivar, 'coordinates',
!     .                         .NOT.do_warn, ' ', maxlen, attlen,
!     .                         attoutflag, aux_coord_names, tmp )  
!           IF ( got_it ) THEN
!* parse the list of white-space-separated auxiliary variable names
!	      CALL TM_PARSE_STRING(aux_coord_names,max_names,
!     .				   coord_names,num_names)
!* ... this is a poorly formatted notice, but may never be issued
!	      IF (num_names .GT. max_names) CALL WARN(
!     .		 'netCDF coordinates attribute on variable '//
!     .		  ds_var_code(tmvar)//'has too many names')
!	      i = 1
!	      DO i=1, attlen
!	        IF (
!XXXXX  --INTERRUPTED AT EARLY STAGE OF WRITING THIS CODE
!for each variable that possess a coordinates attribute
!	- loop through the coordinate variables listed
!	- identify the orientation of each
!        - put Ferret attributes onto the parent variable pointing to
!              larerz target, time2d target, etc.
!        - set ds_its_aux_coord on the target variable
!          ENDIF
!after the loop, if an aux variable was targetted as the 2dtime variable
!then create TF_CAL_T and TF_LAG_F axes (move this logic from CD_SCAN_VARS) 

* !! TEMPORARILY!!! identify the 2d time variable from the attribute
* _ CoordinateAxisType
*  Eventually we will recognize it from an assemblage of facts.
*   o other variables point to it as a coordinate
*   o it is 2d in the tf plane

	   IF ( STR_CASE_BLIND_COMPARE(vname(:vlen), 'time').EQ.str_eq 
     .    .AND. nvdim .EQ. 2 ) THEN  
              maxlen = 16  
              got_it = NC_GET_ATTRIB( dset, ivar, '_CoordinateAxisType',
     .                            do_warn, vname(:vlen), maxlen, attlen,
     .                            attoutflag, name, vals ) 
              IF ( got_it
     .       .AND. STR_CASE_BLIND_COMPARE(name, 'Time').EQ.str_eq ) THEN
* ... Synthesize a 1D time axis and forecast lead time for the colection
*     Also modify the native time axis to become a true delta (lag) time
	         CALL CD_1D_LINE_FROM_2D (dset, cdfid, ivar, vname,
     .					  axcalt, axlagf, native_tax, status)
	      ENDIF
	   ENDIF
	ENDDO

*               ... Now set a global attribute for this dataset with the
*                   new axis name
* TBD

* successful completion
 	status = merr_ok
        RETURN

* error exit
 5900   RETURN
	END
