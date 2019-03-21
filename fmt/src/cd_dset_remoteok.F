	  SUBROUTINE CD_DSET_REMOTEOK(dset, ok_remote)


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

C Check whether we can define a LET/REMOTE variable in this dataset.
* Sets the flag  ds_accepts_remote for the dataset.

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'

	INTEGER dset
	LOGICAL ok_remote

	LOGICAL TM_HAS_STRING
	INTEGER TM_LENSTR1, slen, vlen, ivar, cdfid, cdfstat, status
	CHARACTER vname*128, buff*3000, ebuff*3000

* Already checked? 
	IF (ds_accepts_remote(dset)) GOTO 5000

* If not an opendap set, then it does not accept remote definitions
	IF ( ds_des_name(dset)(1:7) .NE. 'http://') GOTO 5100

	DO 100 ivar = 1, maxvars
	   IF (ds_var_setnum(ivar) .EQ. dset) GOTO 110
 100	CONTINUE
        GOTO 5000	 !no variables in the datset, so return FALSE
 110	CONTINUE

	slen = TM_LENSTR1(ds_des_name(dset))

* If its already a datset with f-tds variable in it, it cannot be used.
* (depending on where we're calling this, there should be a message).

	IF (TM_HAS_STRING(ds_des_name(dset)(1:slen), 'letdeq1')) GOTO 5000

* Name of first variable, use it to define a variable name.

	vname = ds_var_code(ivar)
	vlen = TM_LENSTR1(vname)

* Construct a url which is the original url, with a variable defined,
* LET varname_1_NEW = 7
	
	buff = '{}{letdeq1 '//vname(:vlen)//'1_new=7}'
	slen = TM_LENSTR1(buff)

* Encode the variable definition, set up the full url, 
* and try to open it.

	CALL CD_ENCODE_URL ( buff, ebuff, status)

	slen = TM_LENSTR1(ds_des_name(dset))
	vlen = TM_LENSTR1(ebuff)
	buff = ds_des_name(dset)(:slen)//'_expr_'//ebuff(:vlen)

	cdfstat = NF_OPEN(buff, NF_NOWRITE, cdfid) 

* If it could be opened, we can make remote-variable definitions.

	IF (cdfstat .EQ. NF_NOERR) THEN
	   ds_accepts_remote(dset) = .TRUE.

* close the url with f-tds var - we were just testing here.
	   cdfstat = NF_CLOSE( cdfid )

	ELSE
	   ds_accepts_remote(dset) = .FALSE.
	   CALL WARN ('dataset does not accept REMOTE variable definitions')
	ENDIF

 5000	ok_remote = ds_accepts_remote(dset)
	RETURN

 5100	CONTINUE
	ds_accepts_remote(dset) = .FALSE.
	CALL WARN ('dataset does not accept REMOTE variable definitions')
	END

