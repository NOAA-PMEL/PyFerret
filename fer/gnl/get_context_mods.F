	SUBROUTINE GET_CONTEXT_MODS
     .		      ( dflt_cx,
     .			text_buff,
     .			mods_cx,
     .			max_list_len,
     .			list_len,
     .			it_start,
     .			it_end,
     .			report_unknown,
     .			status )

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
* interpret a list of coordinate limit specifiers and other special qualifiers
* and put the interpreted list of modifications into the buffer mods_cx
* note: no initialization is done except on those dimensions that are modified

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/25/86
* revision 1.0 - 3/2/87  - changed to "/n=lo:hi@XXX" syntax
* revision 1.1 - 3/20/87 - changes for new TMAP library
* revision 1.2 - 4/28/87 - allowed X=@NUL syntax and "relative" notation
* revision 1.3 - 6/25/87 - if x=@AVE and y=@AVE then do AAV ("area average")
*			 - also corrected bug for qualifiers with first letter
*			   of x,y,z,t,i,j,k,l,d or g
* revision 1.4 - 8/18/87 - added decoding of transformation argument and special
*			   check that @SHF is applied only to subscripts
* revision 1.5 - 8/29/87 - added GFDL_VMS.PARM
* revision 1.6 - 9/14/87  - oddness check for @SBX
*			    and reverse order of lo>hi instead of error
*			    and /X with no position is ignored
* revision 1.7 - 9/23/87  - allow 4 character region names from cx_name
* revision 1.8 - 3/3/88   - forbid @SHF with world coords to avoid loosing
*			    transformation in FINISH_1_CONTEXT_MO*
* revision 1.9 - 3/23/88  - added processing of date/time strings
* revision 2.0 - 7/25/88  - added /G= decoding (requiring dflt_cx)
*			    and /D=dset_name (instead of number)
* revision 2.1 - 9/28/88  - bug fix: re-ordering date string test wrong
* revision 2.2 -11/22/88  - new io library FIND_GRID_NUMBER -->TM_GET_GRIDNUM
* revision 2.3 - 1/4/89   - added regridding transformations
*			  - corrected bugs in VAR[Q=xxx] (Q syntax) and @FAV
* revision 2.4 - 3/14/89  - distinguish between X=160E:160W and X=160W:160E
*			    using "formatted" flag in TRANSLATE_LIMIT
* revision 2.5 - 4/24/89  - bug fix:  iend --> namelen
* V200:  6/28/89 - eliminated %%_relative, added grid_needs_dset
*	10/20/89 - check for x=lo::hi (colon2_pos error)
*	 2/16/90 - more detailed error reporting
* Unix/RISC port - 1/91 - cant use "//" with CHAR*(*) variable
*                - 8/91 added Hanning,Partzen and Welch smoothers to err msg
* V230: 1/21/92 - coach user to use SHOW TRANSFORMS for transform errors
*       8/10/92 - allow multiple specifications of same axis eg u[@zlim,z=@trn]
*       8/22/92 - incorporated GX=grid@trn regridding notation (major chges)
*               - fixed improper error messages using risc_buff
*               - do not impose area average regridding h--> GET_CONTEXT_GRID
*               - trap G=pseudo-var error
* V320:12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V400: 3/20/95 - ignore any region info that contains grave accents.  This
*		  is because this routine is called **during** the process
*		  of evaluating the grave accent expressions.
* V402: 7/95	- Allow nested [] of the form vname1[G=vname2[D=dset]]
* V420: 9/95	- Support for implicit grid (e.g. var[gx=axis_name]
*	    Note: This change also altered slightly the interpretation of the
*		  regridding qualifiers.  "gx=salt" now means to pull only
*		  the X axis from the variable SALT --> an implicit grid
* 	(IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"
*	1/9/96  - (for modulo regridding) decode regrid transform names
*		  in excess of 3 characters
*	2/13/96 - allow GX=lo:hi:delta equivalent to X=lo:hi:delta
*	2/23/96 - fix bugs left in check added 8/10/92 that miscode
*		   [Y=@DDF,J=1] --> added "set_implicitly" in addition to
*		   already existing "first_time"
* V430:  7/1/96 - incorporate 4D transformations: AVE, VAR, DIN, NDG, NBD
*	7/10/96 - allow axes of unknown orientation to be used in a GX=axis
* Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V450:  7/17/97 - GX=grid_name does not generate X-only regridding
*       8/97 - *kob* - set cx_calendar to show whether a time was entered
*                      as a date string.
*                    - also check cx_calender rather than fmat_lo or fmat_hi
* V491:  3/2/98 *sh* - increased length of "name" to accomodate
*                    e.g. gt="1-jan-1990:12:00":"15-jan-1990:12:00":120
*                    and allow special check for leading quotation mark
* V510: 8/99 *sh* - allow g=uvar and return encoded char pos of uvar
*	12/99 *sh* - save info about formatting of inputs in cx_formatted
*	12/99 *sh* - added ugly hack to allow multiple axes as in "GXY=gg"
*	12/99 *sh* bug fix: g=user_var failed if user_var was pure abstract
*	3/00  *sh* - remove old trap for mixed dates and time steps
*	4/00  *sh* - since netCDF uses dynamic grids now, DEALLOC will
*		     deallocate axes within implicit grids. Accomodate that.
* V520: 6/00 *sh* bug fix: cx_grid_needs_dset is not set correctly when the
*		           syntax is X[Gx=var[D=dset]]
* V530: 3/01 *acm* - Pass calendar ID, if available, to TRANSLATE_LIMIT
* V532 5/01 *sh*: minor bug fix: detect extra colon in "i=1:5:2:"
* V533:6/01 *sh* bug fix: cx_grid_needs_dset is not set correctly when the
*			  g=var and var is a uvar with self-contained dset 
*
* *kob* 1/03 - replace call to intrinsic func CHAR with its octal
*              equivalent - needed for g77 port
* specifiers may give coordinate in subscript units (/I,/J,/K,/L) or
* may give coordinate in "world" units (/X,/Y,/Z,/T)
* the specifiers may be of any of the following forms:
*	/q=val			specifies a single value
*	/q=val_1:val_2		specifies a range of values
*	/q=val_1:val_2:val_inc	specifies a range and an increment
*	/q=val_1:val_2@XXX	specifies a range with a transformation XXX
*	/q=@XXX			specifies a transformation XXX, only

* logical arguments:
*	report_unknown 	  if .TRUE. then unknown specifiers are errors

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	include	'xtm_grid.cmn_text'
	external xdset_info_data
	include 'errmsg.parm'	
	include 'ferret.parm'	
	include 'gfdl_vms.parm'	
	include 'interp_stack.parm'	
	include 'xalgebra.cmn'
	include 'xprog_state.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'		! for RISC/FORTRAN

* calling arguments:
	CHARACTER*(*)	text_buff
	INTEGER		mods_cx, status, max_list_len, list_len, ict,
     .			dflt_cx, it_start(max_list_len), it_end(max_list_len)
	LOGICAL		report_unknown

* internal variables:
	LOGICAL		INVALID_R8_TO_I4, TM_DIGIT,
     .			atsin, colon, colon2, atcolon, first_time(4),
     .			kludge_gx_to_x(4), set_implicitly(4),
     .			fmat_lo, fmat_hi, logical_dummy, single_char,
     .			has_g_eq_unkn_uvar, gdset_included
	INTEGER		ATOM_POS, STR_CASE_BLIND_COMPARE, STR_UPCASE,
     .			TM_GET_GRIDNUM, KNOWN_GRID, FIND_DSET_NUMBER,
     .			REGION_NUMBER, TM_GET_LINENUM, TM_AXIS_DIRECTION,
     .			equal_pos, colon_pos, atsin_pos, atcolon_pos, grid,
     .			suborval, lo_start, lo_end, namlen,
     .			hi_start, hi_end, item, idim, del_start, del_end,
     .			colon2_pos, vax_code, istart, iend, itrans,
     .			dset, variable, category, name_end, next_char, i,
     .			impl_grid, axis, other, naxes
	CHARACTER*(1)	UPPER_CASE, region_qual(4,2), other_qual(2),
     .			letter, letter2
	CHARACTER	name*48, comment*12, transnam*6, buff*32
	REAL*8		temp
	INTEGER		dstart, dend, lbrac, state, gdset

* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = O'011' )
	INTEGER		looking_for_D, looking_for_eq, looking_for_name,
     .			done_looking, cal_id, TM_GET_CALENDAR_ID
	PARAMETER     ( looking_for_D = 1,
     .			looking_for_eq = 2,
     .			looking_for_name = 3,
     .			done_looking = 9)

	DATA		(region_qual(idim,1),idim=1,4) / 'I','J','K','L' /,
     .			(region_qual(idim,2),idim=1,4) / 'X','Y','Z','T' /,
     .			 other_qual(1) / 'D' /,	! /DATA_SET=
     .			 other_qual(2) / 'G' /	! /GRID=


* initialize
	dset = cx_data_set( dflt_cx)
        axis = unspecified_int4
        grid = unspecified_int4

        DO 10 idim = 1, 4    ! 8/92
	kludge_gx_to_x(idim) = .FALSE.
	set_implicitly(idim) = .FALSE.
 10     first_time(idim) = .TRUE.
	has_g_eq_unkn_uvar = .FALSE.

* check in advance for the use of pre-specified ("/@") limits and D=
* (the results of D= may be relevant in G=var)
	DO 50 item = 1, list_len
	   istart = it_start(item)
	   iend   = it_end  (item)
	   risc_buff  = text_buff(istart:iend)		! for RISC/FORTRAN
	   len_rbuff  = MIN(size_rbuff, iend-istart+1)	! for RISC/FORTRAN
	   letter = text_buff(istart:istart)
	   IF (letter .EQ. '@') THEN
	      IF ( iend .EQ. istart ) GOTO 5015
* ... pre-specified limit has been requested
	      ict = REGION_NUMBER( text_buff( istart+1:iend) )
	      IF ( ict .EQ. unspecified_int4 ) GOTO 5100
* ... set predefined context limits as defaults
 20	      DO 30 idim = 1,4
	         IF ( cx_given( idim, ict ) ) THEN
	            cx_lo_ss ( mods_cx, idim ) = cx_lo_ss ( ict, idim )
	            cx_hi_ss ( mods_cx, idim ) = cx_hi_ss ( ict, idim )
	            cx_lo_ww ( idim, mods_cx ) = cx_lo_ww ( idim, ict )
	            cx_hi_ww ( idim, mods_cx ) = cx_hi_ww ( idim, ict )
* kob 9/97 - set cx_calendar as well
		    IF (idim .EQ. t_dim) 
     .                 cx_calendar(mods_cx) = cx_calendar(ict)
	            cx_trans ( idim, mods_cx ) = cx_trans   ( idim, ict )
	            cx_trans_arg(idim, mods_cx ) = cx_trans_arg(idim, ict )
	            cx_by_ss ( idim, mods_cx ) = cx_by_ss( idim, ict )
	            cx_given ( idim, mods_cx ) = cx_given( idim, ict )
!                    first_time(idim) = .FALSE.	! pre 2/23/96
                    set_implicitly(idim) = .TRUE.
	         ENDIF
 30	      CONTINUE
	   ELSEIF (letter.EQ.'d' .OR. letter.EQ.'D' ) THEN
	      IF ( iend .EQ. istart ) GOTO 5015
* make sure it isn't some other qualifier that begins with "d"
!	      IF ( iend .GT. istart ) THEN
	         IF (  text_buff( istart+1:istart+1 ) .NE. ' '
     .	         .AND. text_buff( istart+1:istart+1 ) .NE. '='
     .	         .AND. text_buff( istart+1:istart+1 ) .NE. tab ) GOTO 50
!	      ENDIF
* ... D=data_set given
	      equal_pos = ATOM_POS( text_buff(istart:iend), '=' )
	      IF ( equal_pos .EQ. atom_not_found
     .	      .OR. equal_pos .EQ. atom_illegal_pos ) GOTO 5110
	      equal_pos = equal_pos + istart - 1
	      name = text_buff(equal_pos+1:iend)
	      namlen = iend-equal_pos
	      dset = FIND_DSET_NUMBER( name )
	      IF ( dset .EQ. unspecified_int4 ) GOTO 5040
	      cx_data_set( mods_cx ) = dset
	      cx_dset_gvn( mods_cx ) = .TRUE.
	   ENDIF
 50	CONTINUE

* loop through list of other ( non /@ ) items provided
	DO 500 item = 1, list_len
	   istart = it_start( item )
	   iend   = it_end  ( item )
	   risc_buff  = text_buff(istart:iend)		! 8/92
	   len_rbuff  = MIN(size_rbuff, iend-istart+1)	! 8/92
	   letter = UPPER_CASE(text_buff(istart:istart))! 8/92

* ignore any region information that contains grave accents
	   IF ( INDEX(risc_buff,'`') .GT. 0 ) GOTO 500

* all valid specifiers are single characters or double character
* regrid specifiers of the form G*, when * is I,J,K,L,X,Y,Z, or T. Is this ?
	   IF ( iend .GT. istart ) THEN
              letter2 = text_buff(istart+1:istart+1) ! 8/92
	      IF (  letter2 .EQ. ' '
     .	       .OR. letter2 .EQ. '='
     .	       .OR. letter2 .EQ. tab ) THEN
* ... is it a region specifier (single character) ?
                 single_char = .TRUE.
	         naxes = 1
                 DO 54 suborval = 1,2
	         DO 54 idim     = 1,4
	            IF ( letter.EQ.region_qual(idim,suborval) ) GOTO 200
 54              CONTINUE
                 GOTO 100     ! not a region specifier
              ELSE                                    ! 8/92
                 single_char = .FALSE.
                 letter2 = UPPER_CASE(letter2)
                 DO 60 suborval = 1,2
	         DO 60 idim     = 1,4
	            IF ( letter2.EQ.region_qual(idim,suborval) ) GOTO 100
 60 	         CONTINUE
                 GOTO 190   ! ... not a recognized qualifier
              ENDIF
	   ENDIF

* maybe its /D= (data_set) or /G= (grid#)
 100       IF ( letter.EQ.'D' ) GOTO 500 ! already done
	   IF ( letter.NE.'G' ) GOTO 190

* G= or G*= non-standard grid specifier ...
* full format is (e.g.) GX=SALT@AVE (atsin and beyond optional)
* (7/95) or may also include data set for grid as in G=vname[D=dset]
	   equal_pos = ATOM_POS( text_buff(istart:iend), '=' )
	   IF ( equal_pos .EQ. atom_not_found
     .	   .OR. equal_pos .EQ. atom_illegal_pos ) GOTO 5020
	   equal_pos = equal_pos + istart - 1
	   name = text_buff(equal_pos+1:iend)
	   namlen = iend-equal_pos
	   atsin_pos = ATOM_POS( name, '@' )
	   comment = '@ttt syntax'			! ... for error message
	   IF ( atsin_pos .EQ. atom_illegal_pos ) GOTO 5020	   
	   atsin = atsin_pos .NE. atom_not_found
!           IF (.NOT.(single_char.OR.atsin)) GOTO 5026 !8/92 GX=TEMP forbidden
	   IF ( atsin ) THEN
	      name_end = atsin_pos - 1
	   ELSE
	      name_end = namlen
	   ENDIF

* allow multiple axes at once as in "GXY=ggg" (12/99 hack)
	   IF (.NOT.single_char) THEN
* count the number of axis specifiers
	      DO 102 i = istart+1, equal_pos
                 letter2 = text_buff(i:i)
	         IF (  letter2 .EQ. ' '
     .	          .OR. letter2 .EQ. '='
     .	          .OR. letter2 .EQ. tab ) GOTO 103
 102          CONTINUE
! can never get here ...
 103	      naxes = i - istart - 1
	      next_char = istart
	   ENDIF

* initialize for next axis as in "GXY="
 105	   IF (.NOT.single_char) THEN
	      next_char = next_char + 1
	      IF ( next_char-istart .GT. naxes ) GOTO 500
              letter2 = UPPER_CASE( text_buff(next_char:next_char) )
              DO 107 suborval = 1,2
	      DO 107 idim     = 1,4
	         IF ( letter2.EQ.region_qual(idim,suborval) ) GOTO 109
 107	      CONTINUE
              GOTO 190   ! ... not a recognized qualifier
 109	      CONTINUE
	   ENDIF

* decipher the grid name (or variable name) (or axis name - 9/95)
*11111111111
           IF ( name_end .GT. 0 ) THEN
* ... first see if grid has been given as a LOW:HI:DELTA  2/96
*	Note: this trap is a kludge -- it is a hack to allow GX=lo:hi:delta
*	      equivalent to X=lo:hi:delta because it is natural for users
*	      to confuse these two
*       3/98: for T axis with GT="1-JAN-1990" first character may be a quote
	      i = 1		                                     ! 3/98
	      IF (idim .EQ. t_dim .AND. name(1:1) .EQ. '"') i = 2    ! 3/98
	      IF ( TM_DIGIT(name(i:i)) .AND. .NOT.single_char ) THEN
	         IF (.NOT.first_time(idim) ) GOTO 5200
	         IF (naxes .GT. 1) GOTO 5205	! cannot say GXY=lo:hi:delta
	         kludge_gx_to_x(idim) = .TRUE.
	         GOTO 201	! treat it like X=lo:hi:delta
	      ENDIF
* ... see if grid has been given by grid name
              grid = TM_GET_GRIDNUM( name(:name_end) )
*22222222222222
* ... logic for implicit grids added here 7/97
	      IF ( grid .NE. unspecified_int4 ) THEN
	         IF ( single_char ) THEN	! g=grid_name
	            IF ( cx_grid(mods_cx) .NE. unspecified_int4
     .	           .AND. cx_grid(mods_cx) .NE. grid ) GOTO 5022
 	            cx_grid( mods_cx ) = grid
	         ELSE
	            IF (.NOT.cx_has_impl_grid(mods_cx)) THEN
	               CALL ALLO_GRID_TO_CX(mods_cx,impl_grid,status)
	               IF ( status .NE. ferr_ok ) GOTO 5000
	            ENDIF
	            grid_line(idim,impl_grid) = grid_line(idim,grid)
		 ENDIF
	         cx_unstand_grid(mods_cx) = .TRUE.
*22222222222222
	      ELSE
* ... no ... then maybe it's a variable name (or an axis name -- 9/95)
* 	     if "[D=dset]" is not given then assume it is in the default dset
* note: as of FERRET 2.3 g=user_var is still forbidden.  This restriction can
*	be lifted by adding code in IS_USER_VAR to inspect for simultaneous
*	cx_unstand_dset and cx_data_set=unspecified after determining variable
*	or pseudo-variable context.  A new hi_cx has to be set up for the g=
*	user variable and then RETURN to get that component
*    7/95: allow embedded data set info "vname[D=dset]"
	         lbrac = INDEX(name(:name_end),"[")
*33333333333333333
	         IF (lbrac .GT. 0) THEN
* ... find closing right bracket		
	            IF ( lbrac .EQ. name_end ) GOTO 5056
	            dend = INDEX(name(lbrac+1:name_end),"]")
	            IF ( dend .EQ. 0 ) GOTO 5056
	            name_end = lbrac-1		! variable name end
	            dend   = dend + lbrac - 1	! embedded data set name end
* ... parse "[D=dset]"
	            state = looking_for_D
	            DO 112 i = lbrac+1,dend
	               IF ( name(i:i) .EQ. ' '
     .		       .OR. name(i:i) .EQ. tab ) THEN
	                  CONTINUE
	               ELSEIF ( state .EQ. looking_for_D ) THEN
	                  IF ( name(i:i) .EQ. 'd'
     .		          .OR. name(i:i) .EQ. 'D' ) THEN
	                     state = looking_for_eq
	                  ELSE
	                     GOTO 5056
	                  ENDIF
	               ELSEIF ( state .EQ. looking_for_eq ) THEN
	                  IF ( name(i:i) .EQ. '=' ) THEN
	                     state = looking_for_name
	                  ELSE
	                     GOTO 5056
	                  ENDIF
	               ELSE	! 1st non-white character after the "=" 
	                  dstart = i
	                  state = done_looking
	                  GOTO 114
	               ENDIF
 112	            CONTINUE
 114	            IF ( state .NE. done_looking ) GOTO 5056
	            gdset = FIND_DSET_NUMBER( name(dstart:dend) )
	            gdset_included = .TRUE.
*33333333333333333
	         ELSE
	            gdset = dset
	            gdset_included = .FALSE.
*33333333333333333
	         ENDIF
* ... end of 7/95 additions
* ... which variable (from which to borrow grid)?
	         CALL FIND_VAR_NAME( gdset, name(:name_end),
     .                               category, variable )
	         IF ( category .EQ. cat_pseudo_var ) GOTO 5055
*33333333333333333
	         IF ( variable .NE. munknown_var_name ) THEN
	            grid = KNOWN_GRID(gdset,category,variable)
		    cx_grid_needs_dset = .NOT.cx_dset_gvn( mods_cx )
* ... 6/00 bug fix for syntax X[gx=var[d=dset]] -- look for [d=...]
	            IF ( cx_grid_needs_dset )  cx_grid_needs_dset
     .					       = .NOT.gdset_included
* ... 6/01 bug fix for grid from uvar with self-contained data set 
	            IF (cx_grid_needs_dset .AND. category.EQ.cat_user_var)
     .			cx_grid_needs_dset = uvar_need_dset(variable)
* ... 8/99 -- allow G=uvar where uvar may be unknown at this point
		    IF ( grid     .EQ. unspecified_int4
     .		   .AND. category .EQ. cat_user_var ) THEN	! g=??uvar??
	              IF (cx_grid(mods_cx) .NE. unspecified_int4) GOTO 5022
		      has_g_eq_unkn_uvar = .TRUE.
*       HACK!: encode the start/end limits of the uvar name into the grid !!!
		      cx_grid(mods_cx) = unspecified_int4
     .				 - (equal_pos+1 + 2000*iend)
		      IF (cx_has_impl_grid(mods_cx)) THEN
*       A any previous gx= info becomes irrelevant
	                 CALL INIT_GRID( cx_impl_grid(mods_cx),
     .		                        '(implicit)', unspecified_int4 )
		         cx_has_impl_grid(mods_cx) = .FALSE.
			 cx_impl_grid(mods_cx) = unspecified_int4
		         CALL DEALLO_GRID(status)
			 IF (status .NE. ferr_ok) CALL ERRMSG
     .				( ferr_internal, status, 'g=uvar', *5000) 
	              ENDIF
* ... end of 8/99
	            ELSEIF ( single_char ) THEN	! g=name
	               IF ( cx_grid(mods_cx) .NE. unspecified_int4
     .	              .AND. cx_grid(mods_cx) .NE. grid ) GOTO 5022
 	               cx_grid( mods_cx ) = grid
	            ELSEIF (.NOT.has_g_eq_unkn_uvar) THEN
	               IF (.NOT.cx_has_impl_grid(mods_cx)) THEN
	                  CALL ALLO_GRID_TO_CX(mods_cx,impl_grid,status)
	                  IF ( status .NE. ferr_ok ) GOTO 5000
	               ENDIF
	               grid_line(idim,impl_grid) = grid_line(idim,grid)
	            ENDIF
*33333333333333333
	         ELSEIF ( single_char ) THEN
	            GOTO 5050		! (G=axis_name not allowed)
*33333333333333333
	         ELSE
* ... as a last resort maybe it is an axis name (9/95)
	            axis = TM_GET_LINENUM( name(:name_end) )
		    IF ( axis .EQ. unspecified_int4 ) THEN
	               GOTO 5050	! final defeat
	            ELSEIF ( TM_AXIS_DIRECTION(axis) .NE. idim
     .		       .AND. TM_AXIS_DIRECTION(axis) .NE. 0 ) THEN	! 7/96
	               GOTO 5052
	            ELSEIF (.NOT.has_g_eq_unkn_uvar) THEN
	               IF (.NOT.cx_has_impl_grid(mods_cx) ) THEN
	                  CALL ALLO_GRID_TO_CX(mods_cx,impl_grid,status)
	                  IF ( status .NE. ferr_ok ) GOTO 5000
	               ENDIF
	               grid_line(idim,impl_grid) = axis
	            ENDIF
*33333333333333333
	         ENDIF
	         cx_unstand_grid(mods_cx) = .TRUE.
*22222222222222
	      ENDIF
*11111111111
           ENDIF
* ... explicit transformations given ?
	   IF ( atsin ) THEN
	      IF ( namlen-atsin_pos .LT. 3 ) GOTO 5020		! 3+ char trans
	      transnam = ' '
 	      vax_code = STR_UPCASE(transnam,name(atsin_pos+1:namlen))
	      DO 180 itrans = 1, max_regrid_trans
 180	      IF ( transnam .EQ. alg_regrid_code( itrans ) ) GOTO 182
	      comment = '?? transform'
	      GOTO 5020			! unknown regrid transform
 182          IF ( single_char ) THEN   ! apply transform to all axes
                 DO 184 idim = 1, 4
 184             IF ( cx_regrid_trans(idim,mods_cx)
     .           .EQ. unspecified_int4 ) cx_regrid_trans(idim,mods_cx)
     .                                 = itrans
              ELSE                      ! apply transform to specified axis
                 cx_regrid_trans( idim, mods_cx ) = itrans
              ENDIF
              cx_unstand_grid( mods_cx ) = .TRUE.
	   ENDIF
	   IF (.NOT.single_char) GOTO 105	! back for another axis
	   GOTO 500

* can't identify it
 190 	IF ( text_buff(istart:istart)
     .		.EQ. '@') GOTO 500	! "/@anything" already processed
	IF ( report_unknown ) THEN
	   GOTO 5010			! unknown specifier is error
	ELSE
	   GOTO 500			! ignore unknown specifier
	ENDIF

* specified by value or subscript ?
 200	IF ( kludge_gx_to_x(idim) ) GOTO 5200
 201	cx_by_ss( idim, mods_cx ) = suborval .EQ. 1

* syntax check: look for "=", ":" and "@"
* ... "="
	equal_pos = ATOM_POS( text_buff(istart:iend), '=' )
	IF ( equal_pos .EQ. atom_not_found   ) GOTO 500	 ! ignore "/Q" w/out "="
	IF ( equal_pos .EQ. atom_illegal_pos ) GOTO 5010
	equal_pos = equal_pos + istart - 1

* ... find first non-blank following "="
	DO 250 lo_start = equal_pos+1, iend
	   IF ( text_buff(lo_start:lo_start) .NE. ' '
     .	  .AND. text_buff(lo_start:lo_start) .NE. tab ) GOTO 260
 250	CONTINUE
	GOTO 5010		! just "X=" and nothing more

* ... "@"
 260	atsin_pos = ATOM_POS( text_buff(istart:iend), '@' )
	IF ( atsin_pos .EQ. atom_illegal_pos ) GOTO 5130
	atsin	  = atsin_pos .NE. atom_not_found
	atsin_pos = atsin_pos + istart - 1
	IF ( atsin_pos + 3 .GT. iend ) GOTO 5130
	IF ( atsin ) THEN
	   del_end = atsin_pos - 1
	ELSE
	   del_end = iend
	ENDIF

* ... ":"
	colon_pos = ATOM_POS( text_buff( istart:del_end ), ':' )
	IF ( colon_pos .EQ. atom_illegal_pos ) GOTO 5140
	colon  = colon_pos .NE. atom_not_found
	IF ( colon ) THEN
	   colon_pos = colon_pos + istart - 1
	   lo_end	 = colon_pos - 1
	ELSE
	   colon2	 = .FALSE.
	   lo_end	 = del_end
	ENDIF

*  ... second ":"
	IF ( colon ) THEN
	   colon2_pos = ATOM_POS(text_buff( colon_pos+1:del_end ), ':' )
	   IF ( colon2_pos .EQ. atom_illegal_pos
     .	   .OR. colon2_pos .EQ. 1		 ) GOTO 5150
	   colon2 = colon2_pos .NE. atom_not_found
	   IF ( colon2 ) THEN
	      colon2_pos = colon2_pos + colon_pos
	      hi_end     = colon2_pos - 1
	     IF (INDEX(text_buff(colon2_pos+1:del_end),':').GT.0)
     .							GOTO 5150
	   ELSE
	      hi_end     = del_end
	   ENDIF
	   hi_start	 = colon_pos + 1
	   del_start	 = colon2_pos + 1
	ENDIF

*  ... GX=lo:hi is legal ONLY with a delta supplied
	IF ( kludge_gx_to_x(idim) .AND. .NOT.colon2 ) GOTO 5210
 
* ... ":" following @ (as in @SHF:-1) ?
	IF ( atsin ) THEN
	   atcolon_pos = ATOM_POS( text_buff(atsin_pos+1:iend), ':' )
	   IF ( atcolon_pos .EQ. atom_illegal_pos ) GOTO 5160
	   atcolon = atcolon_pos .NE. atom_not_found
	   IF ( atcolon ) atcolon_pos = atcolon_pos + atsin_pos
	ELSE
	   atcolon = .FALSE.
	ENDIF

* make allowances for quotation marks here and there (kludge!)
* watch for empty strings! (10/95)
	IF ( text_buff(lo_start:lo_start) .EQ. '"' ) lo_start = lo_start + 1
	IF ( text_buff(lo_end  :lo_end  ) .EQ. '"' ) lo_end   = lo_end   - 1
	IF ( colon ) THEN
	   IF ( text_buff(hi_start:hi_start) .EQ. '"' ) hi_start = hi_start + 1
	   IF ( text_buff(hi_end  :hi_end  ) .EQ. '"' ) hi_end   = hi_end   - 1
	   IF ( hi_start .GT. hi_end ) GOTO 5165
	ENDIF
	IF ( colon2 ) THEN
	   IF (text_buff(del_start:del_start) .EQ. '"') del_start = del_start+1
	   IF (text_buff(del_end  :del_end  ) .EQ. '"') del_end   = del_end  -1
	   IF ( del_start .GT. del_end ) GOTO 5165
	ENDIF

* two specifications of region in same block of region info?
        IF ( .NOT.first_time(idim) ) THEN  ! 2/23/92
	   GOTO 5220
        ELSEIF ( .NOT.set_implicitly(idim) ) THEN  ! 8/92 modified 2/23/92
* ... initialize this dimension if it's the first reference to it

	   cx_given ( idim, mods_cx ) = .TRUE.
	   cx_delta ( idim, mods_cx ) =  unspecified_val8
        ENDIF

*  See if we can find out the calendar, if grid/axis is known
	IF (grid .EQ. unspecified_int4  .AND.  
     .	    axis .EQ. unspecified_int4)  THEN 
	   cal_id = 1
	ELSE
	   IF (axis .EQ. unspecified_int4) axis = grid_line(t_dim, grid)
	   cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )
	ENDIF

* is there a first value given ?
	IF ( lo_start .GT. lo_end ) THEN
	   fmat_lo = .FALSE.
	ELSE
	   CALL TRANSLATE_LIMIT
     .		      ( text_buff(lo_start:lo_end), idim,
     .			cx_by_ss( idim, mods_cx ),
     .			cx_lo_ww ( idim, mods_cx ),
     .			fmat_lo, cal_id, status )
	   IF ( status .NE. ferr_ok ) RETURN
* kob 8/97 - set cx_calender to true/false depending on whether date is
*            a formated string or not
	   IF (idim.EQ.t_dim) cx_calendar(mods_cx) = fmat_lo
	ENDIF

* is there a second value given ?
	IF ( colon ) THEN
	   CALL TRANSLATE_LIMIT
     .		      ( text_buff(hi_start:hi_end), idim,
     .			cx_by_ss( idim, mods_cx ),
     .			cx_hi_ww ( idim, mods_cx ),
     .			fmat_hi, cal_id, status )
	   IF ( status .NE. ferr_ok ) RETURN
* ... formatted longitudes: add 360 degrees if necessary
	   IF ( idim .EQ. x_dim
     .	 .AND.  ( fmat_hi .OR. fmat_lo )
     .   .AND.  cx_hi_ww(idim,mods_cx) .LE. 
     .				cx_lo_ww(idim,mods_cx) ) THEN
	      IF ( cx_hi_ww(idim,mods_cx) .LE. 
     .				cx_lo_ww(idim,mods_cx) - 360.D0 ) THEN
	         GOTO 5030
	      ELSE
	         cx_hi_ww(idim,mods_cx)
     .			 = cx_hi_ww( idim, mods_cx ) + 360.D0
	      ENDIF
	   ENDIF
!	ELSEIF ( first_time(idim) ) THEN   ! 8/92 ELSE --> ELSEIF
	ELSEIF ( lo_start .LE. lo_end ) THEN ! replaced 8/92 check 2/23/96
! ... 2/23/96 set hi=lo iff a lo value, only, was just given
	   cx_hi_ww( idim, mods_cx ) = cx_lo_ww( idim, mods_cx )
	   fmat_hi = fmat_lo
	ENDIF	

* save the formatting information (12/99)
	cx_formatted(idim) = fmat_lo .OR. fmat_hi
	
* is there a third value given ?  (possibly implying a dynamic axis)
	IF ( colon2 ) THEN
	   CALL TRANSLATE_LIMIT
     .		      ( text_buff(del_start:del_end), idim,
     .			cx_by_ss( idim, mods_cx ),
     .			cx_delta   ( idim, mods_cx ),
     .			logical_dummy, cal_id, status )
	   IF ( status .NE. ferr_ok ) RETURN

* get the dynamic regridding transformation, e.g. x=lo:hi:del@tran
	   IF ( atsin ) THEN
	      comment = '?? transform'
	      risc_buff = text_buff(atsin_pos:atsin_pos+3)
	      len_rbuff = 4
	      DO 300 itrans = 1, max_regrid_trans
	         vax_code = STR_CASE_BLIND_COMPARE(
     .				text_buff( atsin_pos+1:atsin_pos+3 ) ,
     .				alg_regrid_code(itrans)		) 
	      IF ( vax_code .EQ. vms_str_success )GOTO 310
 300	      CONTINUE
	      GOTO 5020			! unknown regridding transformation
 310	      cx_regrid_trans(idim,mods_cx) = itrans
	   ENDIF

	ELSE
* get the axis transformation, e.g. x=lo:hi@tran
	   IF ( atsin ) THEN
	      DO 320 itrans = 1, num_transforms
	         vax_code = STR_CASE_BLIND_COMPARE(
     .				text_buff( atsin_pos+1:atsin_pos+3 ) ,
     .				alg_trans_text( itrans )		) 
	      IF ( vax_code .EQ. vms_str_success )GOTO 330
 320	      CONTINUE
	      GOTO 5170	! unknown transformation
 330	      cx_trans( idim, mods_cx ) = alg_trans_num( itrans )
!	   ELSEIF ( first_time(idim) ) THEN   ! 8/92 set default first time
	   ELSE				      ! replaced 8/92 check 2/23/96
	      cx_trans( idim, mods_cx ) = trans_no_transform
	   ENDIF

* get the argument for the transformation, if given
	   IF ( atcolon ) THEN
	      buff = text_buff(atcolon_pos+1:iend)
	      READ ( buff,* , ERR=5180 ) cx_trans_arg( idim, mods_cx )
	   ENDIF

* make sure the argument - explicit or implicit - is correct
	   IF ( atsin ) THEN
	      CALL DFLT_TRANS_ARG( idim, mods_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5035
	   ENDIF
	ENDIF
	
* make sure low limit is less than high
* ... date strings: stored negatively as seconds since BC
* 8/97 *kob* check cx_calendar rather than if fmat_hi or fmat_lo is set to 
*            see if time is a date
	IF (  idim .EQ. t_dim .AND. cx_calendar(mods_cx) ) THEN
	   IF ( cx_hi_ww(idim,mods_cx) .GT. 
     .				cx_lo_ww(idim,mods_cx) ) THEN
	      temp = cx_lo_ww( idim, mods_cx )
	      cx_lo_ww( idim, mods_cx ) = cx_hi_ww( idim, mods_cx )
	      cx_hi_ww( idim, mods_cx ) = temp
	   ENDIF
* ... all other limits
	ELSEIF ( cx_hi_ww(idim,mods_cx) .LT. 
     .				cx_lo_ww(idim,mods_cx) ) THEN
	   temp = cx_lo_ww( idim, mods_cx )
	   cx_lo_ww( idim, mods_cx ) = cx_hi_ww( idim, mods_cx )
	   cx_hi_ww( idim, mods_cx ) = temp
	ENDIF

* if coordinates were given by subscript check 'em and convert 'em
	IF ( cx_by_ss( idim, mods_cx )
     . .AND. cx_lo_ww(idim,mods_cx) .NE. unspecified_val8 ) THEN
	   IF ( INVALID_R8_TO_I4( cx_lo_ww( idim, mods_cx ) )
     .	.OR. INVALID_R8_TO_I4( cx_hi_ww(idim,mods_cx) )) GOTO 5030
	   IF ( cx_delta( idim, mods_cx ) .NE. unspecified_val8
     .	.AND.INVALID_R8_TO_I4( cx_delta  (idim,mods_cx) )) GOTO 5030
	   cx_lo_ss(mods_cx,idim) = NINT( cx_lo_ww(idim,mods_cx) )
	   cx_hi_ss(mods_cx,idim) = NINT( cx_hi_ww(idim,mods_cx) )
	   cx_lo_ww( idim, mods_cx ) = unspecified_val8
	   cx_hi_ww( idim, mods_cx ) = unspecified_val8
* 9/97 *kob* cx_calendar must reflect these changes as well	  
	   IF (idim .eq. t_dim) 
     .                cx_calendar(mods_cx) = .FALSE.
	ENDIF

* SPECIAL CASES for individual axes
* ... @AAV given explicitly
	IF (  cx_trans(idim,mods_cx) .EQ. trans_4d_ave
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_int_def
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_good_pt
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_bad_pt
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_var
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_reduced ) GOTO 5190

* flag this dimension as specified (8/92)
        first_time(idim) = .FALSE.

500	CONTINUE

* SPECIAL CASES for entire context
* If a grid qualifiers wer given then a grid name from at least 1 was mandatory
        IF ( cx_unstand_grid(mods_cx)
     . .AND. .NOT.cx_has_impl_grid(mods_cx)
     . .AND. cx_grid(mods_cx) .EQ. unspecified_int4 ) GOTO 5024

* @AVE or @DIN applied to both X and Y performed as a 2D integral (for COSINE
* factors) 
* 7/96: average, variance, def. int, ngd, and nbg may be 2, 3, or 4D calcs
*     substitute the 4D transform for the one specified
	DO 610 i = 1, num_4d_trans
	   other = 0
	   DO 600 idim = 1, 4	   
	      IF ( cx_trans(idim,mods_cx) .EQ. alg_1d_to_4d(i) ) THEN
	         IF ( other .EQ. 0 ) THEN
	            other = idim		! found 1 axis
	         ELSE
	            cx_trans(other,mods_cx) = alg_4d_trans(i)
	            cx_trans( idim,mods_cx) = alg_4d_trans(i)
	         ENDIF
	      ENDIF
 600	   CONTINUE
 610	CONTINUE

* same is true for indefinite integrals but never been used yet ...
	IF (  cx_trans(x_dim,mods_cx) .EQ. trans_integ_indef
     .	.AND. cx_trans(y_dim,mods_cx) .EQ. trans_integ_indef )
     .		CALL SPLIT_LIST(pttmode_bull, ttout_lun, 
     .		' @IIN on X and @IIN on Y computed as 1D integrals', 49)

	status = ferr_ok
	RETURN

* error exits
* ... clear implicit grids so axes within them don't get deallocated
 5000	IF (cx_has_impl_grid(mods_cx)) CALL INIT_GRID(
     .		cx_impl_grid(mods_cx), '(implicit)',
     .          unspecified_int4 )
	RETURN
 5010	CALL ERRMSG( ferr_syntax, status, risc_buff(:len_rbuff), *5000 )
 5015	risc_buff = text_buff(istart-1:iend)		! for RISC/FORTRAN
	len_rbuff = MIN(48, iend-istart)		! for RISC/FORTRAN
 	CALL ERRMSG( ferr_syntax, status,
     .		'incomplete specification: '//risc_buff(:len_rbuff), *5000 )
 5020	CALL ERRMSG( ferr_regrid, status,
     .		risc_buff(:len_rbuff)//' : '//comment,*5000 )
 5022	CALL ERRMSG( ferr_regrid, status,
     .		risc_buff(:len_rbuff)//' - more than 1 target grid',
     .          *5000 )
 5024	CALL ERRMSG( ferr_regrid, status, 'no target grid specified',
     .          *5000 )
 5030	CALL ERRMSG
     .		( ferr_out_of_range, status, risc_buff(:len_rbuff), *5000 )
 5035	CALL ERRMSG
     .		( status, status, risc_buff(:len_rbuff), *5000 )
 5040	CALL ERRMSG
     .		( ferr_unknown_data_set, status, risc_buff(:len_rbuff), *5000 )
 5050	CALL ERRMSG
     .		( ferr_unknown_grid, status, risc_buff(:len_rbuff), *5000 )
 5052	CALL ERRMSG
     .		( ferr_regrid, status,
     .		'Axis orientation not matching request:'//
     .		risc_buff(:len_rbuff), *5000 )
 5055	CALL ERRMSG( ferr_unknown_grid, status,	risc_buff(:len_rbuff)//
     .			' --> g=user-or-pseudo-var not allowed', *5000 )
 5056	CALL ERRMSG( ferr_syntax, status, 
     .	 'invalid nested brackets: '//risc_buff(:len_rbuff),*5000)
 5060	CALL ERRMSG( ferr_invalid_command, status, 
     .	 'use @SHF only with IJK or L: '//risc_buff(:len_rbuff),*5000)
 5100	CALL ERRMSG( ferr_syntax, status, 
     .	 'unknown region name: '//risc_buff(:len_rbuff),*5000)
 5110	CALL ERRMSG( ferr_syntax, status, 
     .	 'What data set ?: '//risc_buff(:len_rbuff),*5000)
 5120	CALL ERRMSG( ferr_syntax, status, 
     .	 'What grid ?: '//risc_buff(:len_rbuff),*5000)
 5130   CALL ERRMSG( ferr_syntax, status, 
     .	 risc_buff(:len_rbuff)//pCR//
     .	'"@" must precede a valid transformation like "Z=0:50@AVE"'
     .	//pCR//'Use SHOW TRANSFORMS for a list of legal transformations',
     .  *5000)
 5140	CALL ERRMSG( ferr_syntax, status, 
     .	 risc_buff(:len_rbuff)//pCR//
     .	'":" is used incorrectly. A correct example is "Z=0:50".' ,*5000)
 5150	CALL ERRMSG( ferr_syntax, status, 
     .	 risc_buff(:len_rbuff)//pCR//
     .	'":" is used incorrectly. A correct example is "L=1:100:10".',
     .	*5000)
 5160	CALL ERRMSG( ferr_syntax, status, 
     .	 risc_buff(:len_rbuff)//pCR//
     .	'":" is used incorrectly. A correct example is "L=@SBX:5".',
     .	*5000)
 5165	CALL ERRMSG( ferr_syntax, status, risc_buff(:len_rbuff), *5000)
 5170   CALL ERRMSG( ferr_syntax, status, 
     .	 'Unknown transformation:'//risc_buff(:len_rbuff)//pCR//
     .	 'Use SHOW TRANSFORMS for a list of legal transformations',
     .  *5000)
 5180	CALL ERRMSG( ferr_syntax, status, 
     .	 'Invalid transformation argument:'//risc_buff(:len_rbuff),
     .	*5000)
 5190	CALL ERRMSG( ferr_syntax, status, 
     .	 'transformation reserved for FERRET internal use:'//pCR//
     .	 risc_buff(:len_rbuff), *5000)
 5200	CALL ERRMSG( ferr_syntax, status, 
     .	'used GX=lo:hi:delta with X=lo:hi (ditto for Y,Z, or T)'
     .	 //pCR//
     .	 risc_buff(:len_rbuff), *5000)
 5205	CALL ERRMSG( ferr_syntax, status, 
     .	'lo:hi:delta must be given separately for each axis' //pCR//
     .	 risc_buff(:len_rbuff), *5000)
 5210	CALL ERRMSG( ferr_syntax, status, 
     .	'GX=lo:hi:delta with no delta (or ditto for Y,Z, or T)'
     .	 //pCR//
     .	 risc_buff(:len_rbuff), *5000)
 5220	CALL ERRMSG( ferr_syntax, status, 
     .	'multiple specifications on '//ww_dim_name(idim)//' axis'
     .	 //pCR//
     .	 risc_buff(:len_rbuff), *5000)
	END
