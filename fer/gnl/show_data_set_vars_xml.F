	SUBROUTINE SHOW_DATA_SET_VARS_XML( lun, dset )

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
*
* List information about all variables in the data set in xml-style format

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
	INTEGER   TM_LENSTR, TM_LENSTR1, ivar, slen, grid
        CHARACTER TM_FMT*16, show_str*16

 2020   FORMAT ('<var name="', A, '">' )
 2030   FORMAT( '<units>', A, '</units>' )
 2040   FORMAT( '<long_name>', A, '</long_name>' )
 2050   FORMAT( '<history>', A, '</history>' )
 2060   FORMAT( '<_FillValue>', A, '</_FillValue>' )
 2070   FORMAT( '<missing_value>', A, '</missing_value>' )
 2080   FORMAT( '<grid>', A, '</grid>' )
 2090   FORMAT ('</var>')

* describe the variables in the set, including a report on the axes of this grid.

	DO 400 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 400

           IF ( ds_var_code(ivar) .EQ. '-' ) GOTO 400  

* ... output variable code, units, grid, and type

           slen = TM_LENSTR1(ds_var_code(ivar))
           WRITE ( risc_buff, 2020 ) ds_var_code(ivar)(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

           slen = TM_LENSTR(ds_var_units(ivar))
           IF (ds_var_units(ivar)(1:2) .NE. char_init 
     .           .AND. slen.GT.0) THEN
              slen = TM_LENSTR1(ds_var_units(ivar))
              WRITE ( risc_buff, 2030 ) ds_var_units(ivar)(1:slen)  
              CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF

           slen = TM_LENSTR(ds_var_title(ivar))
           IF (ds_var_title(ivar)(1:2) .NE. char_init 
     .           .AND. slen.GT.0) THEN
              slen = TM_LENSTR1(ds_var_title(ivar))
              WRITE ( risc_buff, 2040 ) ds_var_title(ivar)(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF

           slen = TM_LENSTR(ds_var_titl_mod(ivar))
           IF (ds_var_titl_mod(ivar)(1:2) .NE. char_init 
     .           .AND. slen.GT.0) THEN
              slen = TM_LENSTR1(ds_var_titl_mod(ivar))
              WRITE ( risc_buff, 2050 ) ds_var_titl_mod(ivar)(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF

! missing_flag comes from attribute FillValue; 
! bad_flag comes from missing_value

           show_str = TM_FMT(ds_missing_flag(ivar), 16,16,slen)
           WRITE ( risc_buff, 2060 ) show_str(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

           show_str = TM_FMT(ds_bad_flag(ivar), 16,16,slen)
           WRITE ( risc_buff, 2070 ) show_str(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

           IF (ds_var_type(ivar) .EQ. ptype_string ) THEN
              risc_buff = '<datatype>STRING</datatype>'
           ELSE
              risc_buff = '<datatype>FLOAT</datatype>' 
           ENDIF

           grid = ds_grid_number(ivar)
           CALL SHOW_GRID_XML (lun, grid)

	   WRITE ( risc_buff, 2090 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

 400	CONTINUE


	RETURN
	END
