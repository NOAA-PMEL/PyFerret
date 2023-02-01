	SUBROUTINE CD_STORE_DSET ( dset, status)
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
* After a non- netCDF file is open, store dataset, variables, and attribute 
* information in linked list structure.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 

* V600 6/05 *acm* Attribute control.
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* v604 *acm* 7/07 increase length of ds_name, ds_des_name
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* argument definitions:
*       dset    - pointer to TMAP data set
*	status	- return status

* argument declarations:
	INTEGER	      dset, status

* include files
        include 'netcdf.inc'
        include 'tmap_errors.parm'
	include 'tmap_dims.parm'
	include 'xstep_files.cmn_text'
        external xstep_files_data
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdset_info.cd_equiv'

* local variable declarations:
      INTEGER NCF_INIT_OTHER_DSET, TM_LENSTR1

* For saving dataset information in linked list

      INTEGER NCF_GET_DSNUM, llen, slen1, slen2
      PARAMETER (slen1=2048)      ! size of ds_name string
      PARAMETER (slen2=2048)      ! size of ds_des_name string (should this be even longer??)
      INTEGER*1 fhol_dsname(slen1)      ! c-type Hollerith string buffer
      INTEGER*1 fhol_dspath(slen2)

* Save dataset info in GLOBAL_NCdsetList for attribute handling

      llen = TM_LENSTR1( ds_name(dset) )
      CALL TM_FTOC_STRNG( ds_name(dset)(1:llen), fhol_dsname, slen1 )
      llen = TM_LENSTR1( ds_des_name(dset) )
      CALL TM_FTOC_STRNG( ds_des_name(dset)(1:llen), fhol_dspath, slen2 )

      status = NCF_INIT_OTHER_DSET (dset, fhol_dsname, fhol_dspath) 

* successful completion
	
	RETURN
	END
