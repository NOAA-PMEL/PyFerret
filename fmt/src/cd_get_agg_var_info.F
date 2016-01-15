	SUBROUTINE CD_GET_AGG_VAR_INFO (agg_dset, vname, imemb,
     .      vtype, mdset, igrid, iline, nv, status)
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
* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V698 *sh* 1/16 -- added status check/warning, since not checked by callers
*
* Given aggregate datset, variable name in the set, aggregate sequence number,
* return the Ferret datset id, grid number and variable type for the variable 
* in that member set.
*
* for var in aggregate member-dataset:
* for members of the aggregate, imemb (id= sequence number 1,2,3,...
*   igrid    Ferret grid number
*   iline    Ferret line number of aggregate dimension
*   vtype    1=file-var, 3=user-var
*   mdset    Ferret dataset number
*   nv       sequence number in ds_var_code or uvar_name_code

      INCLUDE 'tmap_errors.parm'


* Argument declarations

	CHARACTER*(*) vname
	INTEGER agg_dset, imemb, vtype, mdset, igrid, iline, nv, status

* Local variable declarations
	INTEGER NCF_GET_AGG_VAR_INFO, varid

	CALL CD_GET_VAR_ID (agg_dset, vname, varid, status)
	status = NCF_GET_AGG_VAR_INFO( agg_dset, varid, imemb, vtype, 
     .		mdset, igrid, iline, nv)

	IF (status .NE. pcd_ferr_ok) CALL WARN('ERROR in CD_GET_AGG_VAR_INFO')
	RETURN
	END
