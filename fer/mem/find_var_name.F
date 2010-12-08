	SUBROUTINE FIND_VAR_NAME ( dset_num, test_name, category, variable )

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
* identify the given name by finding it in one of the known categories

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/4/86
* revision 0.1 - 1/20/87 - changed nvari_prenamed to num_fils_vars for PSI
* revision 0.2 - 3/6/87  - introduced user var search and munknown_var_name
* revision 0.3 - 3/23/87 - added new TMAP library: time axes, file var. names
* revision 0.4 - 3/10/88 - added cat_named_var and unspecified data set
* V200:  6/20/89 - eliminated "named" variables
*	 1/25/90 - padded cvar_name_code with 4 blanks avoid QADXT=QADX match
* V402:  7/95 - Added support for  LET/D=dset var = ...
* V495: 7/7/98 - *sh* allow case-sensitive variable names in quotes from dsets
*       user variable names are allowed in quotations just for
*       consistency. Pseudo-variable names in quotes are not allowed -- so they
*       don't get confused
* V510 5/00 *sh* - don't match on a cvar unless the data set indicates
*	Philander-Seigel
* V554 1/04 *acm* Add category cat_counter_var
*  ywei 05/04 change MATCH_NAME and MATCH_QUOTED_NAME function calls to speed up
* V600 *acm* 6/05 attribute control.  If the variable contains a dot (e.g. varname.attname
*                 or varname.attnum) then find and use just the varname.
* V600 9/05 *acm* Fix bug 1336: 
*                 Restore the order of searching through the different lists
*                 of variable names (the chunk of code now at DO 250).  Yonghua had
*                 moved this for efficiency but it wound up finding the wrong
*                 variable def for user variables defined with LET/D.
* v600 6/06 *acm* initialize category to 0 (needed for solaris)
* V63  *acm* 10/09 Changes for gfortran build
*      *kms* 11/10 add examination of pyvar (cat_pystat_var) variables

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xcalc_vars.cmn'
	include	'xalgebra.cmn'
	include	'xdset_parms.cmn'
        include 'xcontrol.cmn'
        include 'errmsg.parm'
        include 'xpyvar_info.cmn'

* calling argument declarations:
	INTEGER		dset_num, category, variable
	CHARACTER*(*)	test_name

* internal variable declarations:
	INTEGER		TM_LENSTR, ivar, uvdset, len_test, len_mchars,
     .                  j, num_indices, varid, dset, status
	LOGICAL		MATCH_QUOTED_NAME, single_dset, do_err
        CHARACTER       varname*512, attname*128
        CHARACTER*258   buff

* initialize
        buff = test_name
	single_dset = dset_num .NE. unspecified_int4
        len_test = TM_LENSTR(test_name)
        category = 0
        dset = dset_num

        IF (INDEX(test_name, '.') .GT. 0) THEN
           do_err = .TRUE.
           CALL BREAK_VARATTNAME ( test_name, dset, varname, 
     .                             attname, varid, do_err, status )
           IF (status .EQ. ferr_ok) THEN
              category = cat_attrib_val
              RETURN

* String marked as not being a varname.attname, perhaps a string in
* quotes that may contain a dot.

           ELSE IF (status .EQ. ferr_not_attribute) THEN
              GOTO 500 

           ELSE 
              variable = munknown_var_name
              RETURN
           ENDIF
        ENDIF
*

*                  CALL tm_log( test_name, len_test) 

* start by checking counter variable names
*	DO 10 ivar = 1, num_counter_vars
*           len_mchars = TM_LENSTR(countervar_name(ivar))
*	   IF ( MATCH_NAME( test_name, len_test,
*        1	   countervar_name(ivar),len_mchars ) ) THEN
*	      category = cat_counter_var
*	      variable = ivar
*	      RETURN
*	   ENDIF
* 10	CONTINUE

 500 	CALL string_array_find(countervar_name_head,
     .                         test_name,
     .                         len_test,
     .                         string_array_result(1),
     .                         max_uvar,
     .                         num_indices)
        IF(num_indices.GT.0) THEN
              ivar = string_array_result(1)
	      category = cat_counter_var
	      variable = ivar
              test_name = buff
	      RETURN
        ENDIF

* check pseudo-variables names
*	DO 20 ivar = 1, num_pseudo_vars
*           len_mchars = TM_LENSTR(alg_pvar(ivar))
*	   IF ( MATCH_NAME( test_name, len_test, alg_pvar(ivar),len_mchars)) 
*        1	   THEN
*	      category = cat_pseudo_var
*	      variable = ivar
*	      RETURN
*	   ENDIF
* 20	CONTINUE

	CALL string_array_find(alg_pvar_head,
     .                         test_name,
     .                         len_test,
     .                         string_array_result(1),
     .                         max_uvar,
     .                         num_indices)
        IF(num_indices.GT.0) THEN
              ivar = string_array_result(1)
	      category = cat_pseudo_var
	      variable = ivar
              test_name = buff
	      RETURN
        ENDIF

* check data user-defined variables
*       CALL tm_log(my_cbuff, 255)
        CALL string_array_find_quoted(uvar_name_code_head,
     .                                test_name,
     .                                len_test,
     .                                string_array_result(1),
     .                                max_uvar,
     .                                num_indices)
*         cat_test = -1
*         var_test = -1
* ... first check for an explicitly named data set
*	   DO 40 ivar = 1, max_uvar
*	      IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 40
*	      IF ( uvar_dset( ivar ) .NE. dset ) GOTO 40
*              len_mchars = TM_LENSTR(uvar_name_code(ivar))
*	      IF ( MATCH_QUOTED_NAME(test_name,len_test, uvar_name_code(ivar), len_mchars))
*     .        	   THEN
*	         category = cat_user_var
*	         variable = ivar
*	         RETURN
*	      ENDIF
* 40	   CONTINUE
	IF ( single_dset ) THEN
           DO 40 j = 1, num_indices
              ivar = string_array_result(j)
	      IF ( uvar_dset( ivar ) .NE. dset ) GOTO 40
              IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 40
              category = cat_user_var
	      variable = ivar
              test_name = buff
	      RETURN
 40	      CONTINUE


	ELSE
* ... check for an implicitly named data set - the "current" data set
	   uvdset = cx_data_set(cx_last)
	   IF ( uvdset .NE. unspecified_int4 ) THEN
*	      DO 60 ivar = 1, max_uvar
*	         IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 60
*	         IF ( uvar_dset( ivar ) .NE. uvdset ) GOTO 60
*                 len_mchars = TM_LENSTR(uvar_name_code(ivar))
*	         IF ( MATCH_QUOTED_NAME( test_name,len_test, uvar_name_code(ivar),len_mchars))
*     .        	      THEN
*	            category = cat_user_var
*	            variable = ivar
*	            RETURN
*	         ENDIF
* 60	      CONTINUE
           DO 60 j = 1, num_indices
              ivar = string_array_result(j)
	      IF ( uvar_dset( ivar ) .NE. uvdset ) GOTO 60
              IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 60
              category = cat_user_var
	      variable = ivar
              test_name = buff
	      RETURN
 60	      CONTINUE
	   ENDIF
	ENDIF

* ... check the global user variable definitions defined without /D
*	DO 100 ivar = 1, max_uvar
*	   IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 100
*	   IF ( uvar_dset( ivar ) .NE. unspecified_int4 ) GOTO 100
*           len_mchars = TM_LENSTR(uvar_name_code(ivar))
*	   IF ( MATCH_QUOTED_NAME( test_name,len_test,uvar_name_code(ivar),len_mchars) )
*     .        	THEN
*	      category = cat_user_var
*	      variable = ivar
*	      RETURN
*	   ENDIF
* 100	CONTINUE
           DO 100 j = 1, num_indices
              ivar = string_array_result(j)
	      IF ( uvar_dset( ivar ) .NE. unspecified_int4 ) GOTO 100
              IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 100
              category = cat_user_var
	      variable = ivar
              test_name = buff
	      RETURN
 100	      CONTINUE

*       Check pystat variables
        CALL string_array_find_quoted(pyvar_code_head,
     .                                test_name,
     .                                len_test,
     .                                string_array_result(1),
     .                                maxpyvars,
     .                                num_indices)
        IF ( single_dset ) THEN
*           Check for an explicitly named data set
            DO 150 j = 1,num_indices
                ivar = string_array_result(j)
                IF ( pyvar_dset_number(ivar) .NE. dset) GOTO 150
                IF ( pyvar_ndarray_obj(ivar) .EQ. 0 ) GOTO 150
                category = cat_pystat_var
                variable = ivar
                test_name = buff
                RETURN
 150        CONTINUE
        ELSE
*           Check for an implicitly named data set - the "current" data set
            uvdset = cx_data_set(cx_last)
            IF ( uvdset .NE. unspecified_int4 ) THEN
                DO 160 j = 1,num_indices
                    ivar = string_array_result(j)
                    IF ( pyvar_dset_number(ivar) .NE. dset) GOTO 160
                    IF ( pyvar_ndarray_obj(ivar) .EQ. 0 ) GOTO 160
                    category = cat_pystat_var
                    variable = ivar
                    test_name = buff
                    RETURN
 160            CONTINUE
            ENDIF
        ENDIF
*       Check pystat variables without an associated dataset
        DO 170 j = 1,num_indices
            ivar = string_array_result(j)
            IF ( pyvar_dset_number(ivar) .NE. unspecified_int4 ) GOTO 170
            IF ( pyvar_ndarray_obj(ivar) .EQ. 0 ) GOTO 170
            category = cat_pystat_var
            variable = ivar
            test_name = buff
            RETURN
 170    CONTINUE

* check file variables from this data set
*	DO 200 ivar = 1, maxvars
*	   IF ( single_dset .AND. ds_var_setnum( ivar ) .NE. dset ) GOTO 200
*           len_mchars = TM_LENSTR(ds_var_code(ivar))
*	   IF ( MATCH_QUOTED_NAME( test_name,len_test, ds_var_code( ivar ),len_mchars)) 
*        1    THEN
*	      category = cat_file_var
*	      variable = ivar
*	      RETURN
*	   ENDIF
* 200	CONTINUE

        CALL string_array_find_quoted(ds_var_code_head,
     .                                test_name,
     .                                len_test,
     .                                string_array_result(1),
     .                                max_uvar,
     .                                num_indices)
        DO 200 j = 1, num_indices
           ivar = string_array_result(j)
	   IF ( single_dset .AND. ds_var_setnum( ivar ) .NE. dset ) GOTO 200
 	   category = cat_file_var
	   variable = ivar
           test_name = buff
           RETURN
 200	CONTINUE


* ... check the global user variable definitions defined with /D
*       CALL tm_log(my_cbuff, 255)
        CALL string_array_find_quoted(uvar_name_code_head,
     .                                test_name,
     .                                len_test,
     .                                string_array_result(1),
     .                                max_uvar,
     .                                num_indices)
*	DO 250 ivar = 1, max_uvar
*	   IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 250
*	   IF ( uvar_dset( ivar ) .NE. pdset_irrelevant ) GOTO 250
*           len_mchars = TM_LENSTR(uvar_name_code(ivar))
*	   IF ( MATCH_QUOTED_NAME( test_name,len_test,uvar_name_code(ivar),len_mchars))
*     .        	THEN
*	      category = cat_user_var
*	      variable = ivar
*	      RETURN
*	   ENDIF
* 250	CONTINUE
           DO 250 j = 1, num_indices
              ivar = string_array_result(j)
	      IF ( uvar_dset( ivar ) .NE. pdset_irrelevant ) GOTO 250
              IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 250
              category = cat_user_var
	      variable = ivar
              test_name = buff
	      RETURN
 250	      CONTINUE

* check table of calculatable variables (padded to 8 characters)
*   but only if this is a Philander-Seigle model output
	IF ( dset.GE.1 .AND. dset.LE.maxdsets ) THEN
	   IF (dp_phil_sieg(dset).OR.dp_cox(dset)) THEN
	      DO 300 ivar = 1, num_calc_vars
                 len_mchars = TM_LENSTR(cvar_name_code(ivar))
	         IF (MATCH_QUOTED_NAME(test_name,len_test,
     .               cvar_name_code(ivar)//'    ',len_mchars))
     .        	   THEN
	            category = cat_calc_var
	            variable = ivar
                    test_name = buff
	            RETURN
	         ENDIF
 300	      CONTINUE
	   ENDIF
	ENDIF

* not found
	variable = munknown_var_name
        test_name = buff

	RETURN

	END
