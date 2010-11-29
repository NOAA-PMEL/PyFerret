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
* V5.80 13-Sep-2004 *acm* Add data-set specific user variables to the listing
*       30-Sep-2004 *acm* Move listing of dataset info to show_data.F, show any
*                         dataset-specific user-defined variables, use the
*                         tag <var name> rather than <vname>
*       13-Oct-2004 *acm* for consistent XML, write <var name="VNAME" />

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
        CHARACTER uvarname*150

 2040   FORMAT ('<var name="', A, '" />' )

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


* show the data-set-specific user-defined variables

	DO 600 ivar = 1, max_uvar
	   IF (uvar_num_items(ivar) .EQ. uvar_deleted  ) GOTO 600
	   IF (uvar_dset(ivar) .EQ. dset) THEN

* ... output variable name

             uvarname = uvar_name_code(ivar)
             slen = TM_LENSTR1(uvarname)
             WRITE ( risc_buff, 2040 ) uvarname(1:slen)
             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

         ENDIF
 600  CONTINUE

      RETURN
      END
