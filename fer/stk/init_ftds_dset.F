	SUBROUTINE INIT_FTDS_DSET ( dset, uvar, status )

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
* v6.85 *acm* Ferret with LET/REMOTE
* Reset the url for the dataset with its f-tds variable definitions.

        IMPLICIT NONE
* calling argument declarations:
	INTEGER		dset, uvar, status

* local variable declarations
	INTEGER TM_LENSTR1, nvars, nv, slen, vlen, varid, ivar
	CHARACTER buff*3000, ebuff*3000, vname*128

* Includes
	include 'tmap_errors.parm'
        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
        include 'xdset_info.cmn_text'
        external xdset_info_data

	IF (rvar_on_server(uvar)) RETURN

	buff = '_expr_{}{'
	slen = TM_LENSTR1(buff)
	nvars = 0

	DO 100 ivar = 1, maxvars
	   IF (uvar_remote(ivar) .AND. rvar_dset(ivar) .EQ. dset) THEN

* Concatenate multiple definitions with _cr_ not with ;
	      IF (nvars .GT. 0) 
     .	        buff = buff(:slen)//'_cr_'
	      slen = TM_LENSTR1(buff)

	      vname = uvar_name_code(ivar)
	      vlen = TM_LENSTR1(vname)
              buff = buff(:slen)//'letdeq1 '//vname(:vlen)//'='

	      slen = TM_LENSTR1(buff)
	      vlen = TM_LENSTR1(uvar_text(ivar))
              buff = buff(:slen)//uvar_text(ivar)(:vlen)
	      slen = TM_LENSTR1(buff)

	      nv = ivar
	      nvars = nvars + 1

	   ENDIF
  100	CONTINUE

	IF (nvars .EQ. 0) GOTO 5100
	
* Close the url

	buff = buff(:slen)//'}'
* Encode the definitions
	CALL CD_ENCODE_URL ( buff, ebuff, status)
	IF (status .NE. ferr_ok) GOTO 5200
	
* Next make the full url and try to open it.

	slen = TM_LENSTR1(ds_des_name(dset))
	vlen = TM_LENSTR1(ebuff)
	buff = ds_des_name(dset)(:slen)//ebuff(:vlen)


* nc_close the dataset. We're going to re-open it with the
* F-TDS variable definitions

	CALL CD_NCCLOSE (dset, status)

* Reopen dataset with the f-tds variables. Save its cdfid in
* the storege for this dset.

	CALL CD_NCOPEN (buff, dset, status)

	IF (status .NE. ferr_ok) THEN  ! Just reopen the original dataset
	   buff = ds_des_name(dset)
	   CALL CD_NCOPEN (buff, dset, status)
	   status = ferr_remotevar_error
	   GOTO 5000
	ENDIF

* Now will get varids and stuff.

	DO 200 ivar = 1, maxvars
	   IF (uvar_remote(ivar) .AND. rvar_dset(ivar) .EQ. dset) THEN

	      vname = uvar_name_code(ivar)
	      CALL CD_NF_GET_VARID ( dset, vname, varid, status )
	      IF (status .NE. ferr_ok) GOTO 5300

	      rvar_varid (ivar) = varid
	      rvar_on_server (ivar) = .TRUE.  ! Definition in current url.
	      IF (ivar .GE. nv) GOTO 300
	   ENDIF
  200	CONTINUE

  300	CONTINUE

 5000	RETURN
 5100   goto 5000 ! ? need message - no remote vars for this dataset
 5200   buff = 'Error encoding F-TDS url for LET/REMOTE definitions '
        CALL ERRMSG(ferr_remotevar_error, status, buff, *5000 )

 5300   CONTINUE !  need error handling - error getting varid
	buff = 'Error getting remote variable id for LET/REMOTE definition'
        CALL ERRMSG (ferr_remotevar_error, status, buff, *5000 )

	END
