	SUBROUTINE EZ_INIT_DSET (filename,variables,columns,dset,status)
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
* Subroutine takes arguments from call and turns it into an entry in the dataset
* and stepfile subroutines. This will allow access to ASCII and unformatted non-
* TMAP data files. This sets up a bare bones entry. All other initialization
* occurs with EZ_MOD_DSET and EZ_READ.
*
* revision 0.00 - 08/03/89
* revision 1.00 - 08/11/89 - Removed calculations involving items that are not
*			     included on SET DATA command to EZ_READ command.
* revision 1.10 - 10/27/89 - *sh* added columns
* revision 1.11 - 01/25/90 - *sh* changed the default format to "FREE"
* Unix/RISC port 3/11/91 *sh*: filename parsing changed (and longer names)
* note: also changed VMS version to save filename extension as well as name
*       (to differentiate snoopy.lis from snoopy.dat)
* rev 1.2 *sh* 7/92 - check for valid, non-duplicated variable names
* rev 1.3 *sh* 11/92 - allow 24 character variable names
* 3/12/93(4?) - lifted restriction on max number of columns
* 12/94 - lifted restriction that # columns must be multiple of # variables
*	  to accomodate /FORMAT=STREAM
* *sh* 3/2/95 - informative messages issued through TM_NOTE (for GUI)
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
	CHARACTER*4	ez_rev
	PARAMETER 	(ez_rev = '1.11')
*
* Arguments:
*	filename	- name of data file
*	variables	- names of variables (and "-" for skipped variables)
*	columns		- number of columns to read (or int4_init)
*	dset		- (output) data set number
*	status		- (output)
*
	CHARACTER*(*)	filename, variables
	INTEGER*4	columns, dset, status
*
* Include files
#ifdef unix
	include 'xio.parm'
#include "gt_lib.parm"
	include 'ez_lib.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xstep_files.cmn_text'
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:XIO.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:EZ_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
#endif
*
* Local definitions
        LOGICAL         TM_LEGAL_NAME
	CHARACTER*24	var_names(maxezvars)
	INTEGER*4	set_cnt, bpos, epos, ncnt, var_pos, num_names,
     .			nvar, jcnt, stp_cnt, nstp
	CHARACTER	TM_STRING*13, buff*48
*
* Assign data set number, exiting if maximum number of data sets are open
	DO 100 set_cnt = 1,maxdsets
	  IF (ds_name(set_cnt) .EQ. char_init40) THEN
	    dset = set_cnt
	    ds_des_name(dset) = filename
* Parse input line to obtain just filename without directory or extension
            epos = LEN(filename)
#ifdef unix
* Unix: find final pathname slash
            DO 77 bpos = epos, 1, -1
 77         IF ( filename(bpos:bpos) .EQ. PathSeparator ) GOTO 78
            bpos = 0    ! no slash
 78         bpos = bpos + 1
#else
* VMS: find the "]" separating the directory from the filename
	    bpos = INDEX(filename,']') + 1
! changed 3/11/91	    epos = bpos + INDEX(filename(bpos:),'.') - 2
#endif
	    ds_name(dset) = filename(bpos:epos)
	    GOTO 110
	  ENDIF
  100	CONTINUE
* Too many data sets are open
	GOTO 9000
*
* Load Format record
  110   ds_type(dset)		= '  EZ'
	ds_format(dset)		= '  1A'
	ds_source_class(dset)	= ' '
	ds_source(dset)		= ' '
	ds_subsource(dset)	= ' '
	ds_prog_rev(dset)	= ez_rev
*
* Load Background record
	WRITE (ds_expnum(dset),'(A2,I2.2)') 'EZ',dset
	ds_modnum(dset)		= ' '
	ds_title(dset)		= ' '
	ds_mod_title(dset)	= ' '
	ds_forcing_set(dset)	= ' '
	ds_t0time(dset)		= ' '
	ds_ntegrate_tstep(dset)	= 1
	ds_time_run(dset)	= ' '
	DO 120 ncnt = 1,num_aux
  120	ds_aux_modnum(ncnt,dset)= ' '
	ds_parm_name(1,dset)	= 'FORMAT'
	ds_parm_text(1,dset)	= 'FREE'
	DO 130 ncnt = 2,num_parm
	  ds_parm_name(ncnt,dset)	= ' '
	  ds_parm_text(ncnt,dset)	= ' '
  130	CONTINUE
*
* time axis
	ds_time_axis(dset) = unspecified_int4
*
* Load message record
	ds_alert_on_open(dset) 	= .FALSE.
	ds_alert_on_output(dset)= .FALSE.
	ds_message(dset)	= ' '
*
* Break input string into individual variable names
	CALL TM_Parse_string(variables,maxezvars,var_names,num_names)
	IF (num_names .GT. maxezvars) THEN
	  buff = ' '
	  WRITE (buff,1000) num_names
	  CALL TM_NOTE( buff, tooplu ) 
	  buff = ' '
	  WRITE (buff,1002) maxezvars
	  CALL TM_NOTE( buff, tooplu ) 
 1000	  FORMAT( 'attempt to initalize',I3,' variables')
 1002	  FORMAT( 'maximum allowed is',I3,' variables')
	  num_names = maxezvars
	ENDIF
*
* check for illegal and duplicate names (*sh* 7/92)
        DO 155 ncnt = 1,num_names
           IF ( var_names(ncnt) .EQ. '-' ) GOTO 155
           IF ( .NOT.TM_LEGAL_NAME(var_names(ncnt)) ) GOTO 9400
           DO 154 jcnt = 1, ncnt-1
              IF ( var_names(jcnt) .EQ. '-' ) GOTO 154
              IF ( var_names(jcnt) .EQ. var_names(ncnt) ) GOTO 9400
 154       CONTINUE
 155    CONTINUE
*
* Loop through all variable names
	var_pos  = 1
	DO 170 ncnt = 1,num_names
*
* Find the next storage position
	  DO 140 nvar = var_pos,maxvars
	    IF (ds_var_setnum(nvar) .EQ. set_not_open) GOTO 150
  140     CONTINUE
* No room left for new variables
	  GOTO 9100
*
* Load variable record now
  150     var_pos = nvar
	  ds_var_code(nvar)	= var_names(ncnt)
	  ds_var_title(nvar)	= var_names(ncnt)
	  ds_var_titl_mod(nvar)	= ' '
	  ds_var_units(nvar)	= ' '
	  ds_missing_flag(nvar)	= -1.E34
	  ds_bad_flag(nvar)	= -1.E34
	  DO 160 jcnt = 1,4
  160	  ds_ordering(jcnt,nvar)     = jcnt
	  ds_precision(nvar)	= 'SINGLE'
	  ds_aux_set_num(nvar)	= 0
* Other calculated or derived data pertaining to variables
	  ds_ndataperrec(nvar)	= int4_init
	  ds_nrecpervar(nvar)	= int4_init
	  ds_nrecb4var(nvar)	= int4_init
	  ds_var_setnum(nvar)	= dset
	  ds_npospervar(nvar)	= 0
  170	CONTINUE
*
* Save number of variable names (including "-") in data set
	ds_basic_axes(1,dset)	= num_names

* other available storage for later data set mods ...
	ds_basic_axes(2,dset)	= int4_init
	ds_basic_axes(3,dset)	= int4_init
	ds_basic_axes(4,dset)	= int4_init
	ds_basic_axes(5,dset)	= int4_init
	ds_basic_axes(6,dset)	= int4_init

* store number of columns to read
	IF ( columns .EQ. int4_init ) THEN
	   ds_ntuple(dset) 	= num_names
	ELSE
!	   IF ( MOD(columns,num_names).NE.0 ) GOTO 9300    ! removed 12/94
	   ds_ntuple(dset) 	= columns
	ENDIF
*
* Find the next storage position for stepfiles in common
	DO 180 stp_cnt = 1,maxstepfiles
	  IF (sf_setnum(stp_cnt) .EQ. set_not_open) THEN
	    nstp = stp_cnt
	    GOTO 190
	  ENDIF
  180	CONTINUE
* No room left for new stepfiles
	GOTO 9200
*
* Load stepfile record
  190	sf_name(nstp)		= filename
	sf_setnum(nstp)		= dset
	sf_aux_set_num(nstp)	= 0
	sf_nrecperstep(nstp)	= 1
	sf_skip(nstp)		= 0
	sf_regvars(nstp)	= .TRUE.
*
* All done
	status = merr_ok
	GOTO 9999
*
* ERROR MESSAGES
 9000	CALL TM_ERRMSG ( merr_dsetlim, status, 'EZ_INIT_DSET',
     .			 no_descfile, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxdsets)),
     .			 no_errstring, *9950)

 9100	CALL TM_ERRMSG ( merr_varlim, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxvars)),
     .			 no_errstring, *9950)

 9200	CALL TM_ERRMSG ( merr_filim, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxstepfiles)),
     .			 no_errstring, *9950)

! 9300	CALL TM_ERRMSG ( merr_badcolmn, status, 'EZ_INIT_DSET', ! removed 12/94
!     .			 dset, no_stepfile,
!     .			 TM_STRING(DBLE(columns)),
!     .			 no_errstring, *9950)

 9400	CALL TM_ERRMSG ( merr_syntax, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 'illegal or duplicate variable name '
     .                   //var_names(ncnt),
     .			 no_errstring, *9950)

 9950	dset = unspecified_int4

* get out
 9999	RETURN
	END
