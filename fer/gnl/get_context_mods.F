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
* *kob* 2/03 - replace call to intrinsic func CHAR with its octal
*              constant equivalent - needed for g77 port
* V552 3/03 *acm* - variable "name" may be the name of a dataset, 
*                    increase its length from 48 to 512 characters.
* V552 5/03 *acm* - bug fixes for parsing mulit-axis regrids, when uvar is
*                   not loaded [gxy=var]  and [gxy=var[d=dset]]
* v554 11/03*acm* - changing a comment, only. 
* v5.80 *acm* 11/04 use cx_cal_id to store the calendar used when
*                   computing world coordinates.
* v6.12 *acm*  7/08 Fix bug 1585: allow for alternative calendars. See if 
*                   there is a calendar definition for the grid and axis, 
*                   and set the calendar ID before translating the date string 
*                   to an axis coordinate.
* V6.8  *acm* 3/12 remove block that checks for qual_given(slash_T0). Neither 
*                  slash_t0 nor orient are defined here.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V6.90 *sh* 12/13 support for auxiliary regridding variables:
*                  temp[gz(depth)=zax@ave]
*		   including special interpretation of gz(depth)=100 (a point)
*		   ensure that the same axis region isn't set twice (line 200)
*		   Detect 'layerz' and 'curvi' automation flags
*	           Note that errors will be silent in dependency_tree_mode
*		   Set cx_g_eq_regrid logical (used to be local "regrid_dims")
* V6.91+ *acm* 8/14 Fix ticket 2189
* V693+ *sh* 12/14 keep track of regridding using cx_vtree* for ticket #2195
*                  also fix err693_undetected_regrid_dset.jnl
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* V695+ *sh* 2/15 add support for _DQ_ 
* V698+ *sh* 5/15 bug fix: allow blanks before aux varname e.g. "gz( depth)"
* V698+ *sh* 6/15 bug fix: all axes to allow [g*(aux)=pt], not just z_dim
* V698+ *acm* 2/16 Ticket 2362: Change /D= qualifier to /DATASET=
* V751  *acm* 5/19 Ticket 1931; don't try to translate the delta value in LO:HI:DEL
*                  from a formatted string.  It's just a number.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* specifiers may give coordinate in subscript units (/I,/J,/K,/L) or
* may give coordinate in "world" units (/X,/Y,/Z,/T)
* the specifiers may be of any of the following forms:
*	/q=val			specifies a single value
*	/q=val_1:val_2		specifies a range of values
*	/q=val_1:val_2:val_inc	specifies a range and an increment
*	/q=val_1:val_2@XXX	specifies a range with a transformation XXX
*	/q=@XXX			specifies a transformation XXX, only

* this routine also handles the items modifying a var name in square brackets
*        D=dset_or_var
*        Gxyzt = var_grid_or_axis@XXXX
*  ... and added v6.85:
*	 Gxyzt(varx,vary,...)=var_grid_or_axis@XXXX ! regrid with aux vars


* logical arguments:
*	report_unknown 	  if .TRUE. then unknown specifiers are errors

        IMPLICIT NONE
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
	include 'xdependency_tree.cmn'
	include 'xrisc.cmn'		! for RISC/FORTRAN

* calling arguments:
	CHARACTER*(*)	text_buff
	INTEGER		mods_cx, status, max_list_len, list_len, ict,
     .			dflt_cx, it_start(max_list_len), it_end(max_list_len)
	LOGICAL		report_unknown

* internal variables:
	LOGICAL		INVALID_R8_TO_I4, TM_DIGIT, MATCH4,
     .			atsin, colon, colon2, atcolon, first_time(nferdims),
     .			morph_gx_to_x(nferdims), set_implicitly(nferdims),
     .			fmat_lo, fmat_hi, logical_dummy, single_char,
     .			has_g_eq_unkn_uvar, gdset_included, has_auxvar
	INTEGER		ATOM_POS, STR_CASE_BLIND_COMPARE, STR_UPCASE,
     .			TM_GET_GRIDNUM, KNOWN_GRID, FIND_DSET_NUMBER,
     .			REGION_NUMBER, TM_GET_LINENUM, TM_AXIS_DIRECTION,
     .			TM_LENSTR1,
     .			equal_pos, colon_pos, atsin_pos, atcolon_pos, grid,
     .			suborval, lo_start, lo_end, namlen, rgdim,
     .			hi_start, hi_end, item, idim, del_start, del_end,
     .			colon2_pos, vax_code, istart, iend, itrans,
     .			dset, variable, category, name_end, next_char, i,
     .			impl_grid, axis, other, naxes, aux_arg_namlen,
     .			aux_lparen_pos, aux_rparen_pos, aux_arg_count,
     .			iqual,this_dim
	CHARACTER*(1)	UPPER_CASE, region_qual(nferdims,2),
     .			letter, letter2
	CHARACTER	name*512, comment*12, transnam*6, buff*32, 
     .                  cal_name*32, letters*3
	REAL*8		temp
	INTEGER		dstart, dend, lbrac, state, gdset, itest, nend, ii


* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9))
	INTEGER		looking_for_D, looking_for_eq, looking_for_name,
     .			done_looking, cal_id, TM_GET_CALENDAR_ID
	PARAMETER     ( looking_for_D = 1,
     .			looking_for_eq = 2,
     .			looking_for_name = 3,
     .			done_looking = 9)

	DATA (region_qual(idim,1),idim=1,nferdims) / 'I','J','K','L','M','N' /,
     .       (region_qual(idim,2),idim=1,nferdims) / 'X','Y','Z','T','E','F' /
!     .       other_qual(1) / 'D' /,	! /DATA_SET=
!     .       other_qual(2) / 'G' /	! /GRID=
!     .       other_qual(3) / 'S' /	! var[z=100,LAYERZ]
!     .       other_qual(4) / 'C' /	! var[x=140e,Y=0,CURVILINEAR]

* local parameter declarations (as in xeq_define)
	INTEGER	    slash_calendar
	PARAMETER ( slash_calendar = 1 + 15)

* initialize
	dset = cx_data_set( dflt_cx)
        axis = unspecified_int4
        grid = unspecified_int4
        DO 10 idim = 1, nferdims
	   morph_gx_to_x(idim)  = .FALSE.
	   set_implicitly(idim) = .FALSE.
!	   cx_g_eq_regrid(idim) = .FALSE.   ! initialize *inside* of 500 loop 6/2015
	   first_time(idim)     = .TRUE.
 10	CONTINUE
	has_g_eq_unkn_uvar = .FALSE.
	cx_vtree_regrid_ntarget = 0

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
 20	      DO 30 idim = 1, nferdims
	         IF ( cx_given( idim, ict ) ) THEN
	            cx_lo_ss ( mods_cx, idim ) = cx_lo_ss ( ict, idim )
	            cx_hi_ss ( mods_cx, idim ) = cx_hi_ss ( ict, idim )
	            cx_lo_ww ( idim, mods_cx ) = cx_lo_ww ( idim, ict )
	            cx_hi_ww ( idim, mods_cx ) = cx_hi_ww ( idim, ict )
* kob 9/97 - set cx_calendar as well
		    IF (idim .EQ. t_dim .OR. idim .EQ. f_dim) 
     .                 cx_calendar(idim,mods_cx) = cx_calendar(idim,ict)
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
* Match any portion of DATASET=
               
	      equal_pos = ATOM_POS( text_buff(istart:iend), '=' )
	      IF ( equal_pos .EQ. atom_not_found
     .	         .OR. equal_pos .EQ. atom_illegal_pos ) GOTO 50
              namlen = equal_pos-1
	      IF (namlen.GT.1) THEN
	         name = text_buff(istart:istart+namlen-1)
		 IF (.NOT. MATCH4(name, namlen, 'DATA', 4) ) GOTO 50
	      ENDIF
	      
* We just want to go on if there is a grave-accent expression, e.g. 
*  `var,r=dsetnum` inside the "d="
	      do ii = istart+2,iend
	         IF (  text_buff(ii:ii) .EQ. '`') GOTO 50
	      enddo

* ... D=data_set given

	      equal_pos = equal_pos + istart - 1
	      name = text_buff(equal_pos+1:iend)
	      namlen = iend-equal_pos
	      dset = FIND_DSET_NUMBER( name )
	      IF ( dset .EQ. unspecified_int4 ) GOTO 5040
	      cx_data_set( mods_cx ) = dset
	      cx_dset_gvn( mods_cx ) = .TRUE.
	   ENDIF
 50	CONTINUE

* Get calendar type from the command line

        cal_name = 'GREGORIAN'		! default
        name = " "
        cal_id = TM_GET_CALENDAR_ID (cal_name)
	iqual = qual_given( slash_calendar )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      cal_name, status )
	   IF ( status .NE. ferr_ok ) RETURN
           name = cal_name
	ENDIF

* 9/01 CF conventions have adopted the name 360_day for a 360-day year.

	IF (name .EQ. '360') cal_name = '360_DAY' 
	IF (name .EQ. 'D360') cal_name = '360_DAY' 
	IF (name .EQ. 'STANDARD') cal_name = 'GREGORIAN' 
	IF (name(1:6) .EQ. 'COMMON') cal_name = 'NOLEAP' 
	IF (name(1:6) .EQ. '365_DA') cal_name = 'NOLEAP' 
	IF (name(1:6) .EQ. '366_DA') cal_name = 'ALL_LEAP' 
	cal_id = TM_GET_CALENDAR_ID (cal_name)

* loop through list of other ( non /@ ) items provided
	DO 500 item = 1, list_len
	   istart = it_start( item )
	   iend   = it_end  ( item )
	   risc_buff  = text_buff(istart:iend)		! 8/92
	   len_rbuff  = MIN(size_rbuff, iend-istart+1)	! 8/92
	   letter = UPPER_CASE(text_buff(istart:istart))! 8/92
	   has_auxvar = .FALSE.				! 12/13

* ... 6/2015 initialize this inside of 500 loop to allow [gx(xaux)=xax,gy(yaux)=yax]
           DO idim = 1, nferdims
	      cx_g_eq_regrid(idim) = .FALSE.
	   ENDDO

* ignore any region information that contains grave accents
	   IF ( INDEX(risc_buff,'`') .GT. 0 ) GOTO 500

* identify 'LAYERZ' and 'CURVI' qualifiers
	   IF (MATCH4(risc_buff,len_rbuff,
     .			  'LAYE',   4          ) ) THEN
	      cx_aux_given(z_dim,mods_cx) = .TRUE.
	      GOTO 500
	   ELSEIF (MATCH4(risc_buff,len_rbuff,
     .			  'CURV',   4          ) ) THEN
	      cx_aux_given(x_dim,mods_cx) = .TRUE.
	      cx_aux_given(y_dim,mods_cx) = .TRUE.
	      GOTO 500
	   ENDIF

* all other valid specifiers are single characters or double character
* regrid specifiers of the form G*, when * is I,J,K,L,M,N,X,Y,Z,T,E or F.
* Is this ?
	   IF ( iend .GT. istart ) THEN
              letter2 = text_buff(istart+1:istart+1) ! 8/92
	      IF (  letter2 .EQ. ' '
     .	       .OR. letter2 .EQ. '='
     .	       .OR. letter2 .EQ. tab ) THEN
* ... is it a region specifier (single character) ?
                 single_char = .TRUE.
	         naxes = 1
                 DO suborval = 1,2
	         DO idim     = 1,nferdims
	            IF ( letter.EQ.region_qual(idim,suborval) ) GOTO 200
	         ENDDO
		 ENDDO
                 GOTO 100     ! not a region specifier
              ELSE                                    ! 8/92
                 single_char = .FALSE.
                 letter2 = UPPER_CASE(letter2)
                 DO suborval = 1,2
	         DO idim     = 1,nferdims
	            IF ( letter2.EQ.region_qual(idim,suborval) ) GOTO 100
	         ENDDO
		 ENDDO
                 GOTO 190   ! ... not a recognized qualifier
              ENDIF
	   ENDIF

* maybe its /D= (data_set) or /G= (grid#)
 100	   IF ( letter.EQ.'D' ) GOTO 500 ! already done
	   IF ( letter.NE.'G' ) GOTO 190

!* only T and F axes may be simultaneously used in auxvar regridding
!	   IF (cx_unstand_grid(mods_cx) .AND. cx_naux(mods_cx).NE. 0) THEN
!	     name = 'only T and F regrids may be combined when using'
!	     IF (idim.NE.t_dim .AND. idim.NE.f_dim) GOTO 5300
!	     DO rgdim = 1, nferdims
!	        IF (cx_g_eq_regrid(rgdim)) THEN
!	          IF (rgdim.NE.t_dim .AND. rgdim.NE.f_dim) GOTO 5300
!	        ENDIF
!	     ENDDO
!	   ENDIF  

* G= or G*= non-standard grid specifier ...
* full format is (e.g.) GX=SALT@AVE (atsin and beyond optional)
* (7/95) or may also include data set for grid as in G=vname[D=dset]
* (3/13) or may also have aux vars as in Gxy(curvx,curvy)=vname[D=dset]@ave
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
	   nend = name_end

* allow multiple axes at once as in "GXY=ggg" (12/99 hack)
* ... or "GXY(curvx,curvy)=ggg"  (3/13)
	   IF (.NOT.single_char) THEN
* count the number of axis specifiers and identify them
	      DO 102 i = istart+1, equal_pos
                 letter2 = text_buff(i:i)
		 IF (  letter2 .EQ. ' '
     .	          .OR. letter2 .EQ. '='
     .            .OR. letter2 .EQ. '('
     .	          .OR. letter2 .EQ. tab ) THEN
	            GOTO 103
	         ELSE
*   .... identify the axis that is being regridded
                    DO suborval = 1,2
	            DO idim     = 1,nferdims
	               IF ( letter2.EQ.region_qual(idim,suborval) ) THEN
	                  IF (cx_g_eq_regrid(idim)) THEN
	                     name = 'auxiliary axis named more than once in'
	                     GOTO 5300
	                  ELSE
	                     cx_g_eq_regrid(idim) = .TRUE.
	                     GOTO 102
	                  ENDIF
	               ENDIF
		    ENDDO
	            ENDDO
	            GOTO 5400    ! logic corruption -- should never happen
	         ENDIF
 102          CONTINUE
	      GOTO 5400          ! logic corruption -- should never happen
 103	      naxes = i - istart - 1
c	      next_char = istart + naxes - 1
	      next_char = istart
	      has_auxvar = letter2 .EQ. '('
	      this_dim = idim  ! 1/2014 hack cuz idim isn't reliable
	   ENDIF

* if auxiliary regridding variables have been given, parse em here
	   IF (has_auxvar) THEN
* ... first check for valid usage
!	     IF (naxes .GT. 1) THEN
!		name =
!     .		'this Ferret version doesnt support multi aux regrid args' 
!	        GOTO 5300
!	     ENDIF
!	     IF (idim .NE. z_dim
!     .	   .AND. idim .NE. t_dim
!     .	   .AND. idim .NE. f_dim  ) THEN
!		name =
!     .		'only the Z, T, and F axes can be used in' 
!	        GOTO 5300
!	     ENDIF

	     aux_lparen_pos = i
	     aux_rparen_pos = INDEX(text_buff(i:equal_pos), ')' )
	     IF (aux_rparen_pos .EQ. 0) THEN
	        name = 'unpaired parentheses in'
	        GOTO 5300
	     ELSEIF (aux_rparen_pos .LT. 3) THEN
	        name = 'empty parentheses in'
	        GOTO 5300
	     ENDIF
	     aux_rparen_pos = aux_rparen_pos + i - 1
* ... infer the aux argument count in the parens from the number of commas
	     aux_arg_count = 1
	     aux_arg_namlen = 0
	     IF (idim.EQ.t_dim .OR. idim.EQ.f_dim) THEN
	        rgdim = idim
	     ELSE
*    ... x,y arguments must be in collating order as in gxy(argx,argy)=
	        DO rgdim = 1, nferdims
	           IF (cx_g_eq_regrid(rgdim)) GOTO 704   ! should use "EXIT"
	        ENDDO
	     ENDIF
* (5/15) ... skip leading white space
 704	     DO i = aux_lparen_pos+1, aux_rparen_pos
	        IF (text_buff(i:i) .NE. ' ') EXIT
	     ENDDO
	     cx_aux_arg_start(rgdim, mods_cx) = i
	     cx_aux_arg_end  (rgdim, mods_cx) = i-1
	     cx_aux_stat     (rgdim, mods_cx) = paux_stat_used
	     DO 104 i = aux_lparen_pos+1, aux_rparen_pos-1
	        IF (text_buff(i:i) .EQ. ' ') GOTO 104
	        IF (text_buff(i:i) .NE. ',') THEN
*    ... add one character to the name length
	          aux_arg_namlen = aux_arg_namlen + 1
	          cx_aux_arg_end(rgdim,mods_cx) =
     .				cx_aux_arg_end(rgdim,mods_cx) + 1
	        ELSE
*    ... start of a new argument name
	          IF (aux_arg_namlen .EQ. 0) THEN
	            name = 'missing auxiliary variable name in'
	            GOTO 5300
	          ENDIF
	          aux_arg_count = aux_arg_count + 1
*    ... match this arg to its axis from the "gxy" syntax
	          DO rgdim = rgdim+1, nferdims
	             IF (cx_g_eq_regrid(rgdim)) THEN
	                cx_aux_arg_start(rgdim, mods_cx) = i+1
	                cx_aux_arg_end  (rgdim, mods_cx) = i
	                GOTO 104
	             ENDIF
	          ENDDO
*    ... there are more arguments than there were axes named in "gxy"
	          name = 'no. of auxliary variables must match no. of axes in'
	          GOTO 5300
	        ENDIF
 104	     CONTINUE
	     IF (aux_arg_namlen .EQ. 0) THEN
	        name = 'missing auxiliary variable name in'
	        GOTO 5300
	     ELSEIF (aux_arg_count .NE. naxes) THEN
	        name = 'no. of auxliary variables must match no. of axes in'
	        GOTO 5300
	     ENDIF
	     cx_naux(mods_cx) = cx_naux(mods_cx) + naxes
	   ENDIF

* initialize for next axis as in "GXY="
 105	   IF (.NOT.single_char) THEN
	      next_char = next_char + 1
	      IF ( next_char-istart .GT. naxes ) GOTO 500
              letter2 = UPPER_CASE( text_buff(next_char:next_char) )
              DO suborval = 1,2
	      DO idim     = 1,nferdims
	         IF ( letter2.EQ.region_qual(idim,suborval) ) GOTO 109
	      ENDDO
	      ENDDO
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
	      IF ((idim.EQ.t_dim .OR. idim.EQ.f_dim) ) THEN
	         IF (name(1:1) .EQ. '"'   ) i = 2 ! 3/98
	         IF (name(1:4) .EQ. '_DQ_') i = 5 ! 2/15 
	      ENDIF
	      IF ( TM_DIGIT(name(i:i)) .AND. .NOT.single_char ) THEN
	         IF (.NOT.first_time(idim) ) GOTO 5200
	         IF (naxes .GT. 1) GOTO 5205	! cannot say GXY=lo:hi:delta
	         morph_gx_to_x(idim) = .TRUE.
* 1/2014
* ... treat gx=lo:hi:delta@ave as an implicit ax regrid like x=lo:hi:delta@ave
*     similarly treat gz(depth)=0:100:10 as an implicit axis
*     and also gz(depth)=100 as a single point axis
*     but gz(depth)=0:100@ave is a special case where we need to parse the
*     ""@ave" as a regridding transform, before we can behave like x=lo:hi:del
	         this_dim = idim   ! 1/2014 hack cuz idim isn't reliable
	         IF (.NOT.(has_auxvar.AND.atsin)) THEN
	            GOTO 201
	         ELSE
	            GOTO 170
	         ENDIF
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
*    5/03: bug fix: [gxy=var[d=dset]] failed because name_end was reset in this
*                   section, but is needed when we go back for next axis: use nend

	         lbrac = INDEX(name(:name_end),"[")
*33333333333333333
	         IF (lbrac .GT. 0) THEN
* ... find closing right bracket		
	            IF ( lbrac .EQ. name_end ) GOTO 5056
	            dend = INDEX(name(lbrac+1:name_end),"]")
	            IF ( dend .EQ. 0 ) GOTO 5056
	            nend = lbrac-1		! variable name end
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

* What if it's x[d=1,i=1:3]  ?  dend is pointing at the last bracket.
		    ii = dend
	            DO 113 i = dstart+1,dend
		       IF ( name(i:i) .EQ. ',' ) THEN
		          ii = i-1
			  GOTO 115
		       ENDIF
 113	            CONTINUE
 115		    CONTINUE
		    dend = ii

	            gdset = FIND_DSET_NUMBER( name(dstart:dend) )
	            IF (gdset .EQ. unspecified_int4) GOTO 5058   ! added 1/201
	            gdset_included = .TRUE.
*33333333333333333
	         ELSE
	            gdset = dset
	            gdset_included = .FALSE.
*33333333333333333
	         ENDIF
* ... end of 7/95 additions
* ... which variable (from which to borrow grid)?
	         CALL FIND_VAR_NAME( gdset, name(:nend),
     .                               category, variable )
	         IF ( category .EQ. cat_pseudo_var ) GOTO 5055
*33333333333333333
	         IF ( variable .NE. munknown_var_name ) THEN
	            IF (dependency_tree_mode) THEN
* ... bookkeeping for determining the dependency tree (VTREE)
	               cx_vtree_regrid_ntarget = cx_vtree_regrid_ntarget + 1
	               cx_vtree_unknown_grid(cx_vtree_regrid_ntarget)
     .								= .FALSE. 
	               cx_vtree_var_or_start(cx_vtree_regrid_ntarget) 
     .								= variable
	               cx_vtree_cat_or_end  (cx_vtree_regrid_ntarget)
     .								= category
	               cx_vtree_regrid_dset (cx_vtree_regrid_ntarget)
     .								= gdset
	            ENDIF
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
	              IF (cx_grid(mods_cx) .NE. unspecified_int4 .AND.
     .                  naxes .GT. 1) THEN

* ... 3/03 -- Have we already encoded start/end limits of the uvar name on a  
*             previous reference to the same uvar?  Needed for [gxy=uvar]

                         itest = unspecified_int4 - (equal_pos+1 + 2000*iend)
                         IF (cx_grid(mods_cx) .NE. itest ) GOTO 5022
                      ENDIF

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
	            IF (dependency_tree_mode) THEN
	              cx_vtree_regrid_ntarget = cx_vtree_regrid_ntarget + 1
	              cx_vtree_unknown_grid(cx_vtree_regrid_ntarget)
     .							= .TRUE.
	              cx_vtree_var_or_start(cx_vtree_regrid_ntarget)
     .							= equal_pos + 1
	              cx_vtree_cat_or_end  (cx_vtree_regrid_ntarget)
     .							= equal_pos + name_end
	              cx_vtree_regrid_dset (cx_vtree_regrid_ntarget)
     .							= unspecified_int4
	            ELSE
	               GOTO 5050		! (G=axis_name not allowed)
	            ENDIF
*33333333333333333
	         ELSE
* ... as a last resort maybe it is an axis name (9/95)
c	            axis = TM_GET_LINENUM( name(:nend) )
	            axis = TM_GET_LINENUM( name(:name_end) )
		    IF ( axis .EQ. unspecified_int4 ) THEN
	              IF (dependency_tree_mode) THEN
* ... 12/14 hack to let an unknown grid error condition slip through if we
*	are merely evaluating the dependency tree.
	                 cx_vtree_regrid_ntarget = cx_vtree_regrid_ntarget + 1
	                 cx_vtree_unknown_grid(cx_vtree_regrid_ntarget)
     .							= .TRUE.
	                 cx_vtree_var_or_start(cx_vtree_regrid_ntarget)
     .							= equal_pos + 1
	                 cx_vtree_cat_or_end  (cx_vtree_regrid_ntarget)
     .							= equal_pos + name_end
	                 cx_vtree_regrid_dset (cx_vtree_regrid_ntarget)
     .							= unspecified_int4
*       It is normal for LET-defined variables
*	that the grid may remain unknown until the children grids
*	are determined, so hopefully it is OK here, too.  If not then we
*	might try inserting an innocuous grid, such as
!	                IF (cx_grid(mods_cx) .EQ. unspecified_int4)
!     .			    cx_grid(mods_cx) = EZ or ABSTRACT
	              ELSE
	                GOTO 5050		! final defeat 
	              ENDIF
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
 170	   IF ( atsin ) THEN
	      IF ( namlen-atsin_pos .LT. 3 ) GOTO 5020		! 3+ char trans
	      transnam = ' '
 	      vax_code = STR_UPCASE(transnam,name(atsin_pos+1:namlen))
	      IF (cx_naux(mods_cx) .EQ. 0) THEN
* ... identify the ordinary regridding transformation
	         DO 180 itrans = 1, max_regrid_trans
	            IF ( transnam .EQ. alg_regrid_code( itrans ) ) GOTO 182
 180		 CONTINUE
	      ELSE
* ... identify the auxiliary variable regridding transformation
	         DO 181 itrans = max_regrid_trans+1, max_auxregrid_trans
		    IF ( transnam .EQ. alg_regrid_code( itrans ) ) GOTO 182
 181		 CONTINUE
	      ENDIF
	      comment = '?? transform'
	      GOTO 5020			! unknown regrid transform
* ... trap (temporary?) for use of "@AVE" on auxiliary regridding
 182	      IF (itrans .EQ. pauxrgrd_average) THEN
!	         CALL WARN_OCCASIONALLY(
!     .		   'Box-averaging not available for auxiliary variable '//
!     .		   'regridding.'//pCR//
!     .		   '           Using @PLAVE instead, '//
!     .		   'e.g. gz(depth)=0:100@plave', 10, 101)
	         itrans = pauxrgrd_pwlave
	      ENDIF
              IF ( single_char ) THEN   ! apply transform to all axes
                 DO idim = 1, nferdims
                    IF ( cx_regrid_trans(idim,mods_cx)
     .                   .EQ. unspecified_int4 ) 
     .                     cx_regrid_trans(idim,mods_cx)= itrans
		 ENDDO
              ELSE                      ! apply transform to specified axis
                 cx_regrid_trans( idim, mods_cx ) = itrans
              ENDIF
!              cx_unstand_grid( mods_cx ) = .TRUE.
	   ENDIF
! this 201 branch is a hack to allow the syntax "gz(depth)=lo:hi@ave" to
!     use the preceding code to read the transformation
	   IF ( has_auxvar ) THEN
	      IF (morph_gx_to_x(this_dim) ) GOTO 201 ! 1/2014
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

* trap attempts to define the same axis region twice
 200	IF (.NOT.first_time(idim)) GOTO 5220
	IF ( morph_gx_to_x(idim) ) GOTO 5200

* specified by value or subscript ?
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
	IF ( morph_gx_to_x(idim) ) THEN
	    IF ( has_auxvar ) THEN
*         GZ(depth)=pt and gz(depth)=lo:hi@plave are allowed.
*         GZ(depth)=lo:hi is not allowed
             IF (colon .AND. .NOT.colon2 .AND. .NOT.atsin) GOTO 5210
	    ELSE
	      IF (.NOT.colon2)				   GOTO 5210
	    ENDIF
	ENDIF

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
* add _DQ_ support 2/15
	IF (     text_buff(lo_start:lo_start) .EQ. '"' ) THEN
	   lo_start = lo_start + 1
	ELSEIF ( text_buff(lo_start:lo_start) .EQ. '_'
     .      .AND.lo_start+3 .LE. iend                  ) THEN
	   IF (  text_buff(lo_start:lo_start+3) .EQ. '_DQ_' )
     .	   lo_start = lo_start + 4
	ENDIF
 	IF (     text_buff(lo_end  :lo_end  ) .EQ. '"' ) THEN
	   lo_end   = lo_end   - 1
	ELSEIF ( text_buff(lo_end  :lo_end  ) .EQ. '_'
     .     .AND. lo_end-3 .GE. istart                  ) THEN
	   IF (  text_buff(lo_end-3:lo_end  ) .EQ. '_DQ_' )
     .	   lo_end   = lo_end   - 4
	ENDIF
 
	IF ( colon ) THEN
	   IF (     text_buff(hi_start:hi_start) .EQ. '"' ) THEN
	      hi_start = hi_start + 1
	   ELSEIF ( text_buff(hi_start:hi_start) .EQ. '_' 
     .        .AND. hi_start+3 .LE. iend                  ) THEN
	      IF (  text_buff(hi_start:hi_start+3) .EQ. '_DQ_' ) 
     .	      hi_start = hi_start + 4
	   ENDIF
	   IF (     text_buff(hi_end  :hi_end  ) .EQ. '"' ) THEN
	      hi_end   = hi_end   - 1
	   ELSEIF ( text_buff(hi_end  :hi_end  ) .EQ. '_' 
     .         .AND. hi_end-3 .GE. istart                 ) THEN
	      IF (  text_buff(hi_end-3:hi_end  ) .EQ. '_DQ_' ) 
     .	      hi_end   = hi_end   - 4
	   ENDIF
	   IF ( hi_start .GT. hi_end ) GOTO 5165
	ENDIF

	IF ( colon2 ) THEN
	   IF (    text_buff(del_start:del_start  ) .EQ. '"') THEN
	      del_start = del_start+1
	   ELSEIF (text_buff(del_start:del_start  ) .EQ. '_'
     .       .AND. del_start+3 .LE. iend                    ) THEN
	      IF  (text_buff(del_start:del_start+3) .EQ. '_DQ_') 
     .	      del_start = del_start+4
	   ENDIF
	   IF (    text_buff(del_end :del_end    ) .EQ. '"' ) THEN
	      del_end   = del_end - 1
	   ELSEIF (text_buff(del_end :del_end    ) .EQ. '_'
     .     .AND. del_end-3 .GE. istart                      ) THEN
	      IF  (text_buff(del_end-3:del_end   ) .EQ. '_DQ_')
     .	      del_end   = del_end - 4
	   ENDIF
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
        IF (cal_id .EQ. 0) THEN
           IF (grid .EQ. unspecified_int4  .AND.  
     .	       axis .EQ. unspecified_int4)  THEN 
              cal_id = 1
	   ELSE
	      IF (axis .EQ. unspecified_int4) axis = grid_line(t_dim, grid)
	      cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )
	   ENDIF
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

* ACM 11/04 save calendar used to compute world coordinates
           IF (idim .EQ. t_dim .OR. idim .EQ. f_dim) cx_cal_id(mods_cx) = cal_id

* kob 8/97 - set cx_calender to true/false depending on whether date is
*            a formated string or not
	   IF (idim .EQ. t_dim .OR. idim .EQ. f_dim) 
     .			cx_calendar(idim,mods_cx) = fmat_lo
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
* Issue 1931: can't be a formatted string in the third value that 
* would need translating.  It's just delta in the units of the axis.
	IF ( colon2 ) THEN
	   READ ( text_buff(del_start:del_end),* , ERR=5000 ) cx_delta(idim, mods_cx)
	   IF ( status .NE. ferr_ok ) RETURN

* get the dynamic regridding transformation, e.g. x=lo:hi:del@tran
	   IF ( atsin .AND. .NOT.has_auxvar ) THEN
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
* ... but not gz(depth)=0:100@ave, which was handled as an aux var regrid
	   IF ( atsin .AND. .NOT.has_auxvar) THEN
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
	
* 12/13
* interpret gz(depth)=100 (a point) as a regrid to a dynamic axis of length 1
* similarly gz(depth)=0:100@ave is an implied regrid to a single point axis
	IF (has_auxvar) THEN
* ... identify the axis being regridding in this qualifier
	  DO rgdim = 1, nferdims
	     IF (cx_g_eq_regrid(rgdim)) THEN
* set up for dynamic axis by inserting a special cx_delta signal
	        IF (cx_lo_ww(rgdim,mods_cx) .EQ. unspecified_val8) CYCLE
	        IF (cx_delta(rgdim,mods_cx) .NE. unspecified_val8) CYCLE
	        IF (cx_lo_ww(rgdim,mods_cx) .EQ. cx_hi_ww(rgdim,mods_cx)
     .	       .OR. cx_regrid_trans(rgdim,mods_cx) .EQ. pauxrgrd_pwlave)
     .					cx_delta(rgdim,mods_cx) = pauto_1pt_ax
	     ENDIF
	   ENDDO
	ENDIF

* make sure low limit is less than high
* ... date strings: stored negatively as seconds since BC
* 8/97 *kob* check cx_calendar rather than if fmat_hi or fmat_lo is set to 
*            see if time is a date

* 11/03 *acm* Why do we swap the limits when hi > lo?  This means that
*             repeat/i=3:1:1 executes the same as repeat/i=1:3:1 and
*             list var[i=180:10:10] the same as list var[i=10:180:10]

	IF (  (idim.EQ.t_dim .OR. idim.EQ.f_dim) .AND. cx_calendar(idim,mods_cx) ) THEN
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
	   IF (idim .EQ. t_dim .OR. idim .EQ. f_dim) 
     .                cx_calendar(idim,mods_cx) = .FALSE.
	ENDIF

* SPECIAL CASES for individual axes
* ... @AAV given explicitly
	IF (  cx_trans(idim,mods_cx) .EQ. trans_4d_ave
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_int_def
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_good_pt
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_bad_pt
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_sum
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_4d_var
     .  .OR.  cx_trans(idim,mods_cx) .EQ. trans_reduced ) GOTO 5190

* flag this dimension as specified (8/92)
        first_time(idim) = .FALSE.

500	CONTINUE

* SPECIAL CASES for entire context
* If grid qualifiers were given then a grid name from at least 1 was mandatory
        IF ( cx_unstand_grid(mods_cx)
     . .AND. .NOT.dependency_tree_mode
     . .AND. .NOT.cx_has_impl_grid(mods_cx)
     . .AND. cx_grid(mods_cx) .EQ. unspecified_int4 ) GOTO 5024

* @AVE or @DIN applied to both X and Y performed as a 2D integral (for COSINE
* factors) 
* 7/96: average, variance, def. int, ngd, and nbg may be 2, 3, or 4D calcs
*     substitute the 4D transform for the one specified
	DO 610 i = 1, num_4d_trans
	   other = 0
	   DO 600 idim = 1, nferdims	   
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

* TF regridding syntax may be gt(tf_times)=TF_CAL_T,gf(tf_times)=TF_LAG_F in which case
* what appears as two distinct auxiliary variables is really only one
	IF (cx_naux(mods_cx) .GT. 1) THEN
	   IF ( cx_aux_arg_start(t_dim,mods_cx) .NE. unspecified_int4
     .    .AND. cx_aux_arg_start(f_dim,mods_cx) .NE. unspecified_int4 ) 
     .					cx_naux(mods_cx) = cx_naux(mods_cx) - 1
	ENDIF

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
 5058	CALL ERRMSG
     .		( ferr_unknown_data_set, status, name(dstart:dend), *5000 )
 5060	CALL ERRMSG( ferr_invalid_command, status, 
     .	 'use @SHF only with IJK or L: '//risc_buff(:len_rbuff),*5000)
 5100	CALL ERRMSG( ferr_syntax, status, 
     .	 'unknown region name: '//risc_buff(:len_rbuff),*5000)
 5110	CALL ERRMSG( ferr_syntax, status, 
     .	 'What data set ?: '//risc_buff(:len_rbuff),*5000)
 5112	CALL ERRMSG( ferr_syntax, status, 
     .	 'What calendar ?: '//risc_buff(:len_rbuff),*5000)
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
 5300	CALL ERRMSG( ferr_syntax, status, 
     .  name(:TM_LENSTR1(name))//' regrid of style gz(depth)=zax'
     .	 //pCR//
     .	 risc_buff(:len_rbuff), *5000)
 5400	CALL ERRMSG( ferr_internal, status, 
     .  'get_context_mods logic corrptn', *5000)
	END
