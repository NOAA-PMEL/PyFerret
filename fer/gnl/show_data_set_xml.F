	SUBROUTINE SHOW_DATA_SET_XML( lun, dset )

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
* Output the name and list variables in the data set in xml-style format

* programmer - Ansley Manke, based on show_data_set_vars.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 6-Aug-2003

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'ez_lib.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xez_info.cmn_text'
	external xez_info_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xtext_info.cmn'
	include	'xprog_state.cmn'
	include	'xrisc.cmn'

* calling argument declarations:
	INTEGER	lun, dset
	
* local variable declarations:
	INTEGER TM_LENSTR, TM_LENSTR1, ivar, slen


 2010   FORMAT ('<dataset>')
 2020   FORMAT( '<title>', A, '</title>' )
 2030   FORMAT( '<title_mod>', A, '</title_mod>' )
 2040   FORMAT( '<vname>', A, '</vname>' )
 2050   FORMAT ('</dataset>')

* Write data set title

	WRITE ( risc_buff, 2010 )
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
 
        slen = TM_LENSTR1(ds_title(dset))
	WRITE ( risc_buff, 2020 ) ds_title(dset)(1:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

        slen = TM_LENSTR(ds_mod_title(dset))
        IF (slen .GT. 0) THEN
           slen = TM_LENSTR1(ds_mod_title(dset))
           WRITE ( risc_buff, 2030 ) ds_mod_title(dset)(1:slen)
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
        ENDIF

* List the variables in the set

	risc_buff = ' '

	DO 400 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 400
           IF ( ds_var_code(ivar) .EQ. '-' ) GOTO 400
  
* ... output variable code

*           slen = TM_LENSTR1(ds_var_code(ivar))
           CALL string_array_get_strlen1(ds_var_code_head, ivar, slen)
           WRITE ( risc_buff, 2040 ) ds_var_code(ivar)(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

 400	CONTINUE

	WRITE ( risc_buff, 2050 )
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	RETURN
	END
