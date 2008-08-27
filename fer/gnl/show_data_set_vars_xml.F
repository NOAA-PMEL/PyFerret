	SUBROUTINE SHOW_DATA_SET_VARS_XML( lun, dset)

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
* V5.80 13-Sep-2004 *acm* Add data-set specific user variables to the listing
*       13-Oct-2004 *acm* Make the current dataset the default dataset so
*                         we can get all the info on dataset-specific user-defined
*                         variables in all files.
* V6.1  2/2008 *acm* Use attribute structure to get all attributes for the variable.

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
	include	'xcontext.cmn'
        include 'netcdf.inc'

* calling argument declarations:
	INTEGER	lun, dset
	
* local variable declarations:
	INTEGER   TM_LENSTR, TM_LENSTR1, GRID_FROM_NAME_XML,
     .            STR_SAME, 
     .            ivar, slen, llen, grid, cx_expr, dset_save,
     .            varid, vtype, nvdims, vdims(8), nvatts, blen, i, 
     .            len_attbuff, attoutflag, all_outflag, attid, 
     .            attlen, attype, iatt, status
	REAL      vals(10)

        LOGICAL   NC_GET_ATTRIB, got_it, coordvar, newatt

        CHARACTER TM_FMT*16, show_str*16
        CHARACTER uvarname*150, attname*128, attbuff*2048, buff*14

C New form for attribute output, dont implement it yet. (Check the
C Write statement that uses number 2075 when we change to using these.)

c 2020   FORMAT ('<var name="', A, '">' )
c 2025   FORMAT ('<attribute name="definition="', A, '">' )
c 2030   FORMAT( '<attribute name="units" value="', A, '" />' )
c 2040   FORMAT( '<attribute name="long_name" value="', A, '" />' )
c 2050   FORMAT( '<attribute name="history" value="', A, '" />' )
c 2060   FORMAT( '<attribute name="_FillValue" value="', A, '" />' )
c 2070   FORMAT( '<attribute name="missing_value" value="', A, '" />' ) 
c 2075   FORMAT( '<attribute name="', A, '" value="', A, '" />' )
c 2090   FORMAT ('</var>')

 2020   FORMAT ('<var name="', A, '">' )
 2025   FORMAT ('<definition="', A, '">' )
 2030   FORMAT( '<units>', A, '</units>' )
 2040   FORMAT( '<long_name>', A, '</long_name>' )
 2050   FORMAT( '<history>', A, '</history>' )
 2060   FORMAT( '<_FillValue>', A, '</_FillValue>' )
 2070   FORMAT( '<missing_value>', A, '</missing_value>' ) 
 2075   FORMAT( '<', A, '>', A, '</', A, '>' )
 2080   FORMAT( '<grid>', A, '</grid>' )
 2090   FORMAT ('</var>')

* describe the variables in the set, including a report on the axes of this grid.

* temporarily make this the default data set

        dset_save = cx_data_set(cx_last)
        cx_data_set(cx_last) = dset

	DO 400 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 400

           IF ( ds_var_code(ivar) .EQ. '-' ) GOTO 400  

* ... output variable code, units, grid, and type

*           slen = TM_LENSTR1(ds_var_code(ivar))
           CALL string_array_get_strlen1(ds_var_code_head, ivar, slen)
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

c           IF (ds_var_type(ivar) .EQ. ptype_string ) THEN
c                risc_buff = 
c     .         '<attribute name="ferret_datatype" value="STRING" />'
c           ELSE
c                risc_buff = 
c     .         '<attribute name="ferret_datatype" value="FLOAT" />'
c           ENDIF
           
           IF (ds_var_type(ivar) .EQ. ptype_string ) THEN
              risc_buff = '<ferret_datatype>STRING</ferret_datatype>'
           ELSE
              risc_buff = '<ferret_datatype>FLOAT</ferret_datatype>' 
           ENDIF


	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* other attributes

           uvarname = ds_var_code(ivar)
           CALL CD_GET_VAR_ID (dset, uvarname, varid, status)
           CALL CD_GET_VAR_INFO (dset, varid, uvarname, vtype, nvdims,
     .           vdims, nvatts, coordvar, all_outflag, status)

* Data type of the variable in the file...

c           IF (vtype .EQ. ncbyte ) THEN
c              risc_buff = 
c     .         '<attribute name="infile_datatype" value="BYTE" />'
c           ELSE IF (vtype .EQ. ncchar ) THEN
c              risc_buff = 
c     .         '<attribute name="infile_datatype" value="CHAR" />'
c           ELSE IF (vtype .EQ. ncshort ) THEN
c              risc_buff = 
c     .         '<attribute name="infile_datatype" value="SHORT" />'
c           ELSE IF (vtype .EQ. nclong ) THEN
c              risc_buff = 
c     .         '<attribute name="infile_datatype" value="LONG" />'
c           ELSE IF (vtype .EQ. ncfloat ) THEN
c              risc_buff = 
c     .         '<attribute name="infile_datatype" value="FLOAT" />'
c           ELSE IF (vtype .EQ. ncdouble ) THEN
c              risc_buff = 
c     .         '<attribute name="infile_datatype" value="DOUBLE" />'
c           ENDIF
           
           IF (vtype .EQ. ncbyte ) THEN
              risc_buff = '<infile_datatype>BYTE</infile_datatype>'
           ELSE IF (vtype .EQ. ncchar ) THEN
              risc_buff = '<infile_datatype>CHAR</infile_datatype>'
           ELSE IF (vtype .EQ. ncshort ) THEN
              risc_buff = '<infile_datatype>SHORT</infile_datatype>'
           ELSE IF (vtype .EQ. nclong ) THEN
              risc_buff = '<infile_datatype>LONG</infile_datatype>'
           ELSE IF (vtype .EQ. ncfloat ) THEN
              risc_buff = '<infile_datatype>FLOAT</infile_datatype>'
           ELSE IF (vtype .EQ. ncdouble ) THEN
              risc_buff = '<infile_datatype>DOUBLE</infile_datatype>'
           ENDIF


	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* All attributes not already listed.

           DO 200 iatt = 1, nvatts
              CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

              newatt = .TRUE.
              newatt = (STR_SAME(attname, 'units') .NE. 0)      .AND. 
     .              (STR_SAME(attname, 'long_name') .NE. 0)     .AND. 
     .              (STR_SAME(attname, 'history') .NE. 0)       .AND. 
     .              (STR_SAME(attname, 'missing_value') .NE. 0) .AND. 
     .              (STR_SAME(attname, '_FillValue') .NE. 0)

              IF (newatt) THEN
                 blen = 2048  ! len of attbuff
                 got_it = NC_GET_ATTRIB (dset, varid, attname, 
     .                                .TRUE., uvarname, blen, attlen, 
     .                                attoutflag, attbuff, vals)

                 IF (attype .NE. NCCHAR) THEN
                    blen = 1
                    attbuff = ' '
                    DO 150 i = 1, attlen
                       buff = TM_FMT(vals(i), 7, 14, llen)
                       attbuff(blen:blen+llen+1) = buff//' '
                       blen = blen + llen + 1
 150                CONTINUE
                 ELSE 
                    blen = TM_LENSTR1(attbuff)
                 ENDIF

                 slen = TM_LENSTR1(attname)
	         WRITE (risc_buff, 2075) attname(1:slen), 
     .                  attbuff(1:blen)
                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              ENDIF

 200       CONTINUE

           grid = ds_grid_number(ivar)
           CALL SHOW_GRID_XML (lun, grid, unspecified_int4)

	   WRITE ( risc_buff, 2090 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

 400	CONTINUE

* show the data-set-specific user-defined variables

	DO 600 ivar = 1, max_uvar
	   IF (uvar_num_items(ivar) .EQ. uvar_deleted  ) GOTO 600
	   IF (uvar_dset(ivar) .EQ. dset) THEN

* ... output variable code, units, grid, and type
*     Skip intermediate variables by checking uvar_parent

             uvarname = uvar_name_code(ivar)
             IF (uvar_parent(ivar) .NE. 0) GOTO 600
            
             slen = TM_LENSTR1(uvarname)
             WRITE ( risc_buff, 2020 ) uvarname(1:slen)
             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
             slen = TM_LENSTR1(uvar_text(ivar))      
             WRITE ( risc_buff, 2025 ) uvar_text(ivar)(1:slen)

             slen = TM_LENSTR(uvar_units(ivar))
             IF (uvar_units(ivar)(1:2) .NE. char_init 
     .             .AND. slen.GT.0) THEN
                slen = TM_LENSTR1(uvar_units(ivar))
                WRITE ( risc_buff, 2030 ) uvar_units(ivar)(1:slen)  
                CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
             ENDIF

             slen = TM_LENSTR(uvar_title(ivar))
             IF (uvar_title(ivar)(1:2) .NE. char_init 
     .             .AND. slen.GT.0) THEN
                slen = TM_LENSTR1(uvar_title(ivar))
                WRITE ( risc_buff, 2040 ) uvar_title(ivar)(1:slen)
                CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
             ENDIF

! missing_flag comes from attribute FillValue; 
! bad_flag comes from missing_value

             show_str = TM_FMT(uvar_bad_data(ivar), 16,16,slen)
             WRITE ( risc_buff, 2060 ) show_str(1:slen)
             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

             show_str = TM_FMT(uvar_bad_data(ivar), 16,16,slen)
             WRITE ( risc_buff, 2070 ) show_str(1:slen)
             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

             IF (uvar_data_type(ivar,dset) .EQ. ptype_string ) THEN
                risc_buff = 
     .         '<attribute name="ferret_datatype" value="STRING" />'
             ELSE
                risc_buff = 
     .         '<attribute name="ferret_datatype" value="FLOAT" />'
             ENDIF

             grid = GRID_FROM_NAME_XML( uvarname, cx_last, cx_expr, status )

             grid = uvar_grid(ivar,dset)
             IF (grid .NE. unspecified_int4) 
     .          CALL SHOW_GRID_XML (lun, grid, cx_expr)

             WRITE ( risc_buff, 2090 )
             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

         ENDIF
 600  CONTINUE

* Restore the default data set

      cx_data_set(cx_last) = dset_save

      RETURN
      END
