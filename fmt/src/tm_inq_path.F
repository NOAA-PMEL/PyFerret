      CHARACTER*(*)FUNCTION TM_INQ_PATH
     .				( fname, path, def_ext, do_errmsg, status )
*
* Find the full path name of the given file
* failure to find a file in a status of merr_not_mounted
* depending on the logical do_errmsg an error message may be issued

* using this routine is "nearly" OS-independent between VMS and UNIX
* environment variables substitute for VMS logicals
* the interpretation of the default extension is a little different
* ... on UNIX it must be ".xxx" on VMS it can be name.xxx

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS

* revision 0.00 - 10/23/89 - code lifted from TM_INIT_DSET by Mark Verschell
* Unix/RISC FORTRAN port 3/7/91 *sh* - OS-specific file syntaxes
*                          - generalized to promote VMS/Unix similarities
* revision 1.00 - 10/17/91 - <kob>
*                          - seperate path (if it exists) from file name
*                            and pass them to tm_path_list as two unique
*                            variables.
* revison 1.01  - 2/13/92  - increased size of TM_PATH_LIST, path_list to 512
*  *sh* 2/12/93 - corrected bug due to increased path_list (2/13/92) - it was
*		  equivalenced to "risc_buff" which was too short
*
*  *sh* 5/28/93 - if filename begins with $ then translate the environment var
*		  (this is a partial sol'n - doesn't deal with embedded $'s)
*
* IBM port 14apr94 *kob* : IBM will only allow character string lengths
*			    between 1 and 500.
*			 : CHARACTER*(*) variables cannot be used in concatenations,
*			   so I used risc_buff.

* arguments:
*	fname	   - filename - possibly lacking extension
*                             - possibly including explicit path 
*       path       - default logical or environment variable with path
*	def_ext	   - default file extension
*       do_errmsg  - give error message for file-not-found

* Argument definitions
        LOGICAL         do_errmsg
	INTEGER		set_cnt, status
	CHARACTER*(*)	fname, path, def_ext

* Parameter and common files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'		! error codes
	include 'tmap_format/gt_lib.parm'
#ifdef AIX_XLF
	include 'tmap_format/xrisc_buff.cmn'
#endif
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! Error codes
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XRISC_BUFF.CMN'

        INCLUDE '($SSDEF)' 
        INTEGER  LIB$SYS_TRNLOG, log_status
#endif

* local valiable declarations
#ifdef AIX_XLF
        CHARACTER TM_PATH_LIST*500, path_list*500
#else
        CHARACTER TM_PATH_LIST*512, path_list*512
#endif
        INTEGER TM_LENSTR1, plen,flen,i,postslash
!!!! bug 2/93        EQUIVALENCE ( path_list, risc_buff )

* path may be an environment variable (UNIX) with a path list or
* a logical name (VMS).  Failing that, it is ignored.

#ifdef unix
* reset postslash to 0 every time code enters here. If not reset, it causes
* a problem with the relative version numbers, cutting off first character
* of filename.
      postslash = 0
      flen = TM_LENSTR1(fname)

* if a pathname exists, we want to extract it from the file name,  and
* pass them to tm_path_list as two separate, but lovely, entities.
* *** start of 5/93 addition
      IF ( fname(1:1) .EQ. '$' ) THEN
* ... translate "$" environment variable at beginning of filename
	 postslash = INDEX( fname ,'/' ) + 1
	 IF ( postslash .EQ. 1 ) THEN
	    CALL GETENV( fname(2:flen), path_list )
	    postslash = flen + 1       ! entire filename from environment var
	 ELSE
	    CALL GETENV( fname(2:postslash-2), path_list )
	 ENDIF		
	 IF ( path_list .EQ. ' ' ) GOTO 5202
	 plen = TM_LENSTR1( path_list )
* ... transfer "morepath" info in the case of $ENVIRONMENT/morepath/filename
	 DO 5 i = flen-1,postslash+1,-1
	    IF ( fname(i:i) .EQ. '/' ) THEN
	       path_list = path_list(:plen)//'/'//fname(postslash:i-1)
	       plen = TM_LENSTR1( path_list )
	       postslash = i+1
	       GOTO 6
	    ENDIF
 5	 CONTINUE
 6	 IF ( plen .LT. LEN(path_list) ) THEN
            plen = plen + 1                ! mandatory ' ' at end
         ELSE
            GOTO 5100
         ENDIF
* *** end of 5/93 addition
      ELSEIF ( path .EQ. ' ' ) THEN
         path_list = ' '
         plen = 1
      ELSEIF ( INDEX( fname ,'/' ) .GT. 0 ) THEN
         DO 10 i = flen-1,1,-1
            IF ( fname(i:i) .EQ. '/' ) THEN
               postslash = i+1
               GOTO 11
         ENDIF
 10      continue
 11      path_list =   fname(:postslash-2)
         plen = TM_LENSTR1(path_list)
         plen = plen + 1
* Otherwise, use the environment paths.
      ELSE
         CALL GETENV( path, path_list )    ! ' ' if no translation
!	 IF ( path_list .EQ. ' ' ) GOTO 5204 ! an error but we want to allow it
         plen = TM_LENSTR1( path_list )
         IF ( plen .LT. LEN(path_list) ) THEN
            plen = plen + 1                ! mandatory ' ' at end
         ELSE
            GOTO 5100
         ENDIF
      ENDIF

#else
!!! VMS version untested as of 3/91 !!!
* explicit path given ?  usable logical provided ?
      IF ( path .EQ. ' ' ) THEN
         path_list = ' '
         plen = 1
      ELSEIF ( INDEX( fname ,':' ).GT.0
     .    .OR. INDEX( fname ,']' ).GT.0 ) THEN
         path_list = ' '
         plen = 1
      ELSE
         log_status = LIB$SYS_TRNLOG( path, ,path_list )
         IF ( log_status .EQ. ss$_notran ) THEN
            path_list = ' '
         ELSE
            plen = TM_LENSTR1( path )
            path_list = path( :plen )//':'
            plen = plen + 1
         ENDIF
      ENDIF
#endif

* evaluate the path
* Send path and filename seperately to tm_path_list.
      IF (postslash .EQ. 0) postslash = 1
      IF (postslash .LE. flen) THEN
         TM_INQ_PATH = TM_PATH_LIST( path_list(:plen), 
     .                               fname(postslash:),def_ext, 
     .                               do_errmsg, status )
      ELSE   ! this option if entire name is an environment var
         TM_INQ_PATH = TM_PATH_LIST( path_list(:plen), 
     .                               ' ',def_ext, 
     .                               do_errmsg, status )
      ENDIF

      RETURN

* error exits
 5100 CALL TM_ERRMSG (merr_syntax, status, 'TM_INQ_PATH',
     .			unspecified_int4, no_stepfile,
     .			'path list too long',
     .                  no_errstring, *9900)
#ifdef AIX_XLF
 5202 risc_buff = fname
      CALL TM_ERRMSG (merr_syntax, status, 'TM_INQ_PATH',
     .			unspecified_int4, no_stepfile,
     .			'unknown environment variable: '//risc_buff(1:flen),
     .                  no_errstring, *9900)
#else
 5202 CALL TM_ERRMSG (merr_syntax, status, 'TM_INQ_PATH',
     .			unspecified_int4, no_stepfile,
     .			'unknown environment variable: '//fname(1:flen),
     .                  no_errstring, *9900)
#endif
 9900 RETURN
      END
