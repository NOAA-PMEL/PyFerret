	SUBROUTINE SHOW_DATA(dset, status) 
	
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
* 4/2012 *acm*
* taken out of xeq_show.F, execute SHOW DATA
* v675  *acm  4/12 Fix ticket 1934, simpler SHOW ATTRIBUTE syntax.
* V6.8  acm  6/12 Implement mode_6d_lab, set by default. If canceled, the 
*                 various SHOW commnands will not list info for E,F dimensions.
* V685 4/13 *acm* Ability to define ensemble with /HIDDEN
* V691+ 8/14 *acm* Fix ticket 2194: SHOW with /OUTFILE
* V695 *acm*  2/15 For Forecast aggregations, call the ds_type 'FCT'
* V695 *acm*  9/15 Fix #2317: length of name sent to FIND_DSET_NUMBER
* V698 *sh* 1/16 Removed "Ferret-defined" annotation
* V7   *acm*  6/12 Add Union aggregations: DEFINE DATA/AGG/U, ds_type 'UNI'

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xcontext.cmn'
	include 'xinterrupt.cmn'
	include 'xprog_state.cmn'
	include 'xrisc.cmn' 

	INTEGER dset, status

	LOGICAL TM_HAS_STRING, 
     .          sbrief, sfull, svars, sfile, sattr, sxml, clobber, 
     .          append, only_1, un_hide
	INTEGER TM_LENSTR1, TM_LENSTR, FIND_DSET_NUMBER, 
     .		    sho_file, iset, len, len0,
     .		    slash_brief,
     .		    slash_full,
     .		    slash_vars,
     .		    slash_file,
     .		    slash_attr,
     .		    slash_xml,
     .		    slash_memb
	CHARACTER default*60, outstring*2048

	PARAMETER ( slash_brief    = 1 + 1,
     .		    slash_full     = 1 + 2,
     .		    slash_vars     = 1 + 3,
     .		    slash_file     = 1 + 4,
     .		    slash_attr     = 1 + 6,
     .		    slash_xml      = 1 + 5,
     .              slash_memb     = 1 + 11 )

* initialize
	show_lun = ttout_lun
	sxml = .FALSE.
	status = ferr_ok

   	sbrief = qual_given( slash_brief ) .GT. 0 
	sfull  = qual_given( slash_full  ) .GT. 0
	svars  = qual_given( slash_vars  ) .GT. 0   .OR. sfull
 	sfile  = qual_given( slash_memb  ) .GT. 0
     .	    .OR. qual_given( slash_file  ) .GT. 0   .OR. sfull
 	sattr  = qual_given( slash_attr  ) .GT. 0  
        IF (sattr) sbrief = .TRUE.
 	sxml  = qual_given( slash_xml  ) .GT. 0 

	IF (qual_given( slash_file  ) .GT. 0)  CALL WARN
     .		( '/FILES is deprecated.  Use /MEMBERS')

* 	SHOW DATA/FILE=[/APPEND][/CLOBBER][/HIDDEN]
        sho_file = qual_given( slash_show_data_file )
	clobber = qual_given( slash_show_data_clobber ) .GT. 0
	append = qual_given( slash_show_data_append ) .GT. 0

* SHOW DATA/HIDDEN  if ensembles are defined with /HIDE, then
*             SHOW DAT/HIDDEN will show the member datasets.

	un_hide = qual_given( slash_show_data_hidden ) .GT. 0

	IF (sho_file .GT. 0) THEN
	   CALL OPEN_SHOW_FILE (show_lun, sho_file,  
     .                          clobber, append, status)
	   IF (status .NE. ferr_ok ) GOTO 5000
	ENDIF

	only_1 = num_args .GT. 0
	IF ( only_1 ) THEN
	   dset = FIND_DSET_NUMBER( cmnd_buff(arg_start(1):arg_end(1)) )
	   IF ( dset .EQ. unspecified_int4 ) GOTO 5000
	   un_hide = .TRUE.
	ENDIF

        IF (.NOT. sxml) THEN
           WRITE ( risc_buff, 3700 )	! heading line
           CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
        ELSE
           risc_buff = '<datasets>'
           CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
        ENDIF

	risc_buff = ' '

	DO 790 iset = 1, maxdsets
	   IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )

	   IF ( ds_name(iset) .EQ. char_init1024
     .	   .OR. (only_1 .AND. iset .NE. dset) ) GOTO 790

           IF ( ds_hide(iset) .AND. .NOT.un_hide) GOTO 790
           dset = iset

* ... is it the default data set?
	   IF ( iset .EQ. cx_data_set(cx_last) ) THEN
	      default = '(default)'
	      len0 = 9
	   ELSE
	      default = ' '
	      len0 = 1
	   ENDIF
	   IF (dset.NE.unspecified_int4 .AND. dset.NE.pdset_irrelevant) THEN
	   IF ( TM_HAS_STRING(ds_type(iset), 'ENS') ) THEN
	      default = default(:len0)//'  Ensemble aggregation'
	      len0 = TM_LENSTR1( default )
	   ENDIF
	   IF ( TM_HAS_STRING(ds_type(iset), 'FCT') ) THEN
	      default = default(:len0)//'  Forecast aggregation'
	      len0 = TM_LENSTR1( default )
	   ENDIF
	   IF ( TM_HAS_STRING(ds_type(iset), 'UNI') ) THEN
	      default = default(:len0)//'  Union aggregation'
	      len0 = TM_LENSTR1( default )
	   ENDIF
	   ENDIF

	   len = TM_LENSTR1( ds_des_name( iset ) )
           IF (sxml) THEN

* Write data set name
              IF (len0 .EQ. 1) THEN
                 CALL ESCAPE_FOR_XML (ds_des_name(iset), outstring, len)
                 WRITE (risc_buff,3711) outstring(:len)
	      ENDIF
              IF (len0 .EQ. 9) THEN
                 CALL ESCAPE_FOR_XML (ds_des_name(iset), outstring, len)
                 WRITE (risc_buff,3712) outstring(:len)
              ENDIF
	      CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
* Write data set title
              CALL ESCAPE_FOR_XML (ds_title(iset), outstring, len)
	      WRITE ( risc_buff, 2020 ) outstring(1:len)
	      CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

	      len = TM_LENSTR(ds_mod_title(iset))
	      IF (len .GT. 0) THEN
                 CALL ESCAPE_FOR_XML (ds_mod_title(iset), outstring, len)
	         WRITE ( risc_buff, 2030 ) outstring(1:len)
	         CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
              ENDIF

           ELSE
	      WRITE (risc_buff,3710) iset,ds_des_name(iset)(:len),default(:len0)
	      CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
           ENDIF

	   risc_buff = ' '
* ... by default only name and variables are shown
           IF ( sxml) THEN
              IF (svars) THEN
                 CALL SHOW_DATA_SET_VARS_XML (show_lun, iset)
              ELSE
                 CALL SHOW_DATA_SET_XML (show_lun, iset)
              ENDIF
           ELSE
	      IF ( svars       ) CALL SHOW_DATA_SET_PARAMS 
     .                             (show_lun, iset)
	      IF ( .NOT.sbrief ) CALL SHOW_DATA_SET_VARS
     .                             (show_lun, iset, svars, status)
	      IF ( sfile       ) CALL SHOW_DATA_FILES      
     .                             (show_lun, iset)
	      IF ( sattr       ) CALL SHOW_DATA_SET_ATTRS      
     .                             (show_lun, iset)
           ENDIF
           IF (sxml .AND. (.NOT.svars)) THEN
              risc_buff = '</dataset>'
              CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
           ENDIF
	   IF (status .NE. ferr_ok) GOTO 5000
 790	CONTINUE

        IF (sxml .AND. (.NOT.svars)) THEN
           risc_buff = '</datasets>'
           CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
        ENDIF

	IF (sho_file .GT. 0) CLOSE (UNIT = show_lun)


 2020   FORMAT( '<title>', A, '</title>' )
 2030   FORMAT( '<title_mod>', A, '</title_mod>' )
 3700	FORMAT (T6,'currently SET data sets:')
 3710	FORMAT (T2,I4,'> ',A,2X,A)
 3711	FORMAT ('<dataset name="',A,'" default="false">')
 3712	FORMAT ('<dataset name="',A,'" default="true">')

 5000	RETURN
        END
