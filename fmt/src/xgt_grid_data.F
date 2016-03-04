	BLOCK DATA XGT_GRID_DATA

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
* names and sizes of predefined climatological axes

* programmer - mark verschell (from a program by steve hankin)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 05/07/87 - by Mark Verschell for PMEL/TMAP
* revision 0.10 - 07/27/88 - Parameter name change, grid/lisne definition order
*			     swapped, line now defined by box coord, midpt coord
* revision 0.11 - 11/21/88 - variable initialization changed to char_initXX
* revision 0.20 - 11/29/88 - removed NORMAL and UNKNOWN line definitions
* revision 0.21 - 12/12/88 - added next_line_mem_pos initialization
* revision 0.30 - 12/15/88 - SH changed box boundry coordinates for PSYT / PSYU
*			     to correctly reflect location in GFDL model 
* revision 0.31 - 02/17/89 - Fixed next_line_mem_pos error (1 less than correct)
* revision 0.40 - 01/07/91 - eliminated multoply define line_mem for RISC
*			     by defining and using pline_mem_next
* revision 0.41 - 04/24/91 *sh* unix/RISC port: increased size of grid_filename
* revision 0.42 - 7/2/92 *kob* commented out initialization of line_mem
*			       to speed compilation and decrease mem needs.
* *kob* 10/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
* V510 *sh* 3/00 - initialize through line/grid_ceiling -- for dynamic netCDF
*			grid and line management
* V530*acm* 1/01 - add line_cal_name for alternative calendar types.
* V550 *sh* 11/02 - add initialize line_modulo_len
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
* V62 *acm*  2/09 - Save original upper/lowercase spelling of axis names 
*                   in line_name_orig for CANCEL MODE UPCASE
*      4/12 *acm* - 6D Ferret, formatted F axis is direction FI
* V685 *acm* 3/13 - add climatological axes to the pre-defined axes.
* v685 *acm* 4/13 - Let Ferret compute the modulo lengths for clim. axes
* v697 *acm* 12/115 -Missed some initializations, 4D-->6D grids
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic variables, 
* pre-defined axes and grids.

	include 'tmap_dims.parm'
	include 'implicit.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xunits.cmn_text'
	include 'xtm_grid.cmn_text'

	INTEGER		ngrd_def, nlne_def, grids_left, lines_left,
     .			grid_lines_left, i, j,
     .			pline_mem_next, pline_mem_left, outtype

	PARAMETER	(ngrd_def	= 0,
     .			 nlne_def	= 7,
     .			 grids_left	= grid_ceiling-ngrd_def,
     .			 lines_left	= line_ceiling-nlne_def,
     .			 grid_lines_left= nferdims*grids_left,
     .                   pline_mem_next = 102,
     .                   pline_mem_left = maxlinestore-pline_mem_next+1)

	PARAMETER	(outtype	= 6)  ! netcdf type double, output axis type.

* LAST LINE_MEM POSITION + 1
	DATA	next_line_mem_pos  /pline_mem_next/
        DATA    pline_mem_init / pline_mem_next/

* DEFINE LINES

	DATA
     .		line_name	(clmirr)	/ 'MONTH_IRREG'	/,
     .		line_name_orig	(clmirr)	/ 'MONTH_IRREG'	/,
     .		line_subsc1	(clmirr)	/ 2		/,
     .		line_dim	(clmirr)	/ 12		/,
     .		line_units	(clmirr)	/ 'DAYS'	/,
     .		line_unit_code  (clmirr)	/ -4   /,
     .		line_direction	(clmirr)	/ 'TI'		/,
     .		line_regular	(clmirr)	/ .FALSE.	/,
     .		line_modulo	(clmirr)	/ .TRUE.       /,
     .		line_start	(clmirr)	/ unspecified_val8 /,
     .		line_delta	(clmirr)	/ unspecified_val8 /,
     .		line_tunit	(clmirr)	/ 86400 /,
     .		line_t0		(clmirr)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clmirr)	/'GREGORIAN'/
     
	DATA
     .		line_name	(clmreg)	/ 'MONTH_REG'	/,
     .		line_name_orig	(clmreg)	/ 'MONTH_REG'	/,
     .		line_subsc1	(clmreg)	/ unspecified_int4 /,
     .		line_dim	(clmreg)	/ 12		/,
     .		line_units	(clmreg)	/ 'hour'	/,
     .		line_unit_code  (clmreg)	/ -3   /,
     .		line_direction	(clmreg)	/ 'TI'		/,
     .		line_regular	(clmreg)	/ .TRUE.	/,
     .		line_modulo	(clmreg)	/ .TRUE.       /,
     .		line_start	(clmreg)	/ 366. /,
     .		line_delta	(clmreg)	/ 730.485 /,
     .		line_tunit	(clmreg)	/ 3600 /,
     .		line_t0		(clmreg)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clmreg)	/'GREGORIAN'/


	DATA
     .		line_name	(clseas)	/ 'SEASONAL_REG'	/,
     .		line_name_orig	(clseas)	/ 'SEASONAL_REG'	/,
     .		line_subsc1	(clseas)	/ unspecified_int4 /,
     .		line_dim	(clseas)	/ 4		/,
     .		line_units	(clseas)	/ 'HOURS'	/,
     .		line_unit_code  (clseas)	/ -3   /,
     .		line_direction	(clseas)	/ 'TI'		/,
     .		line_regular	(clseas)	/ .TRUE.	/,
     .		line_modulo	(clseas)	/ .TRUE.       /,
     .		line_start	(clseas)	/ 1095.7275 /,
     .		line_delta	(clseas)	/ 2191.455 /,
     .		line_tunit	(clseas)	/ 3600 /,
     .		line_t0		(clseas)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clseas)	/'GREGORIAN'/

	DATA
     .		line_name	(clmgrg)	/ 'MONTH_GREGORIAN' /,
     .		line_name_orig	(clmgrg)	/ 'MONTH_GREGORIAN' /,
     .		line_subsc1	(clmgrg)	/ 26		/,
     .		line_dim	(clmgrg)	/ 12		/,
     .		line_units	(clmgrg)	/ 'days'	/,
     .		line_unit_code  (clmgrg)	/ -4   /,
     .		line_direction	(clmgrg)	/ 'TI'		/,
     .		line_regular	(clmgrg)	/ .FALSE.	/,
     .		line_modulo	(clmgrg)	/ .TRUE.       /,
     .		line_start	(clmgrg)	/ unspecified_val8 /,
     .		line_delta	(clmgrg)	/ unspecified_val8 /,
     .		line_tunit	(clmgrg)	/ 86400 /,
     .		line_t0		(clmgrg)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clmgrg)	/'GREGORIAN'/
     
	DATA
     .		line_name	(clmnol)	/ 'MONTH_NOLEAP' /,
     .		line_name_orig	(clmnol)	/ 'MONTH_NOLEAP' /,
     .		line_subsc1	(clmnol)	/ 50		/,
     .		line_dim	(clmnol)	/ 12		/,
     .		line_units	(clmnol)	/ 'days'	/,
     .		line_unit_code  (clmnol)	/ -4   /,
     .		line_direction	(clmnol)	/ 'TI'		/,
     .		line_regular	(clmnol)	/ .FALSE.	/,
     .		line_modulo	(clmnol)	/ .TRUE.       /,
     .		line_start	(clmnol)	/ unspecified_val8 /,
     .		line_delta	(clmnol)	/ unspecified_val8 /,
     .		line_tunit	(clmnol)	/ 86400 /,
     .		line_t0		(clmnol)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clmnol)	/'NOLEAP'/

	DATA
     .		line_name	(clm360)	/ 'MONTH_360_DAY' /,
     .		line_name_orig	(clm360)	/ 'MONTH_360_DAY' /,
     .		line_subsc1	(clm360)	/ unspecified_int4	/,
     .		line_dim	(clm360)	/ 12		/,
     .		line_units	(clm360)	/ 'days'	/,
     .		line_unit_code  (clm360)	/ -4   /,
     .		line_direction	(clm360)	/ 'TI'		/,
     .		line_regular	(clm360)	/ .TRUE.	/,
     .		line_modulo	(clm360)	/ .TRUE.       /,
     .		line_start	(clm360)	/ 15. /,
     .		line_delta	(clm360)	/ 30. /,
     .		line_tunit	(clm360)	/ 86400 /,
     .		line_t0		(clm360)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clm360)	/'360_DAY'/

	DATA
     .		line_name	(clmall)	/ 'MONTH_ALL_LEAP' /,
     .		line_name_orig	(clmall)	/ 'MONTH_ALL_LEAP' /,
     .		line_subsc1	(clmall)	/ 74	/,
     .		line_dim	(clmall)	/ 12		/,
     .		line_units	(clmall)	/ 'days'	/,
     .		line_unit_code  (clmall)	/ -4   /,
     .		line_direction	(clmall)	/ 'TI'		/,
     .		line_regular	(clmall)	/ .FALSE.	/,
     .		line_modulo	(clmall)	/ .TRUE.       /,
     .		line_start	(clmall)	/ unspecified_val8 /,
     .		line_delta	(clmall)	/ unspecified_val8 /,
     .		line_tunit	(clmall)	/ 86400 /,
     .		line_t0		(clmall)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clmall)	/'ALL_LEAP'/


* 7/99 *kob* add in initialization for line_name and mpsnorm.  
*            needed for v3.4 of netcdf use
	DATA    line_name       (mpsnorm) /' '/
	DATA    line_name_orig  (mpsnorm) /' '/


* initialize the unused grid common locations
	DATA	grid_filename	/maxgridfiles*char_init128/

	DATA	(line_parent(i),i=1,line_ceiling)
     .			/line_ceiling*0/

	DATA	(line_class(i),i=1,line_ceiling)
     .			/line_ceiling*pline_class_basic/

	DATA	(line_name(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*char_init16/
	DATA	(line_name_orig(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*char_init16/

	DATA	(line_subsc1(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*int4_init/

	DATA	(line_dim(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*int4_init/

	DATA	(line_units(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*char_init16/

	DATA	(line_unit_code(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*int4_init/

	DATA	(line_direction(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*char_init02/

	DATA	(line_start(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*real8_init/

	DATA	(line_delta(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*real8_init/

	DATA	(line_regular(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*.FALSE./

	DATA	(line_t0(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*char_init20/

	DATA	 line_cal_name(0)  /'GREGORIAN'/
	DATA	(line_cal_name(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*'GREGORIAN'/

	DATA	(line_shift_origin(i),i=1,line_ceiling)
     .			/line_ceiling*.FALSE./

	DATA	(line_tunit(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*real4_init/

	DATA	(line_modulo(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*.FALSE./

	DATA	(line_modulo_len(i),i=nlne_def+1,line_ceiling)
     .			/lines_left*0.0D0/
     
	DATA	(line_dattype(i),i=1,line_ceiling)
     .			/line_ceiling*outtype/

c comment out to speed compilation and decrease mem needs *kob* 7/92
c	DATA	(line_mem(i),i=pline_mem_next,maxlinestore)
c     .			/pline_mem_left*real8_init/

	DATA	(grid_name(i),i=ngrd_def+1,grid_ceiling)
     .			/grids_left*char_init16/

	DATA	((grid_line(i,j),i=1,nferdims),j=ngrd_def+1,grid_ceiling)
     .			/grid_lines_left*int4_init/

	DATA	(grid_rotation(i),i=ngrd_def+1,grid_ceiling)
     .			/grids_left*real4_init/

	DATA	((grid_out_prod(i,j),i=1,nferdims),j=ngrd_def+1,grid_ceiling)
     .			/grid_lines_left*.FALSE./

* DEFINE LINE AND BOX COORDS NEXT TO LAST

* MONTH_IRREG

*  COORDS FOR MONTH_IRREG
	DATA	(line_mem(i),i=2,13)
     .	/     15.5, 45.12125, 74.7425, 105.2425, 135.7425, 166.2425, 
     .	     196.7425, 227.7425, 258.2425, 288.7425, 319.2425, 349.7425/

*  BOX BOUNDARIES FOR MONTH_IRREG
	DATA	(line_mem(i),i=14,26)
     .	/     0.0, 31.0, 59.2425, 90.2425, 120.2425, 151.2425, 
     .	    181.2425, 212.2425, 243.2425, 273.2425, 304.2425, 
     .	    334.2425, 365.2425/
     
* MONTH_GREGORIAN
     
*  COORDS FOR MONTH_GREGORIAN
	DATA	(line_mem(i),i=27,38)
     .	/     15.5, 45.12125, 74.7425, 105.2425, 135.7425, 166.2425, 
     .	     196.7425, 227.7425, 258.2425, 288.7425, 319.2425, 349.7425/

*  BOX BOUNDARIES FOR MONTH_GREGORIAN
	DATA	(line_mem(i),i=39,51)
     .	/     0.0, 31.0, 59.2425, 90.2425, 120.2425, 151.2425, 
     .	    181.2425, 212.2425, 243.2425, 273.2425, 304.2425, 
     .	    334.2425, 365.2425/

* MONTH_NOLEAP
     
*  COORDS FOR MONTH_NOLEAP
	DATA	(line_mem(i),i=52,63)
     .	/     15.5,  45.0,  74.5, 105.0, 135.5, 166.0, 196.5, 
     .	     227.5, 258.0, 288.5, 319.0, 349.5/

*  BOX BOUNDARIES FOR MONTH_NOLEAP
	DATA	(line_mem(i),i=64,76)
     .	/     0.0, 31.0, 59.0, 90.0, 120.0, 151.0, 181.0, 
     .      212.0, 243.0, 273.0, 304.0, 334.0, 365.0/

* MONTH_ALL_LEAP
     
*  COORDS FOR MONTH_ALL_LEAP
	DATA	(line_mem(i),i=77,88)
     .	/     15.5, 45.5, 75.5, 106.0, 136.5, 167.0, 197.5, 
     .       228.5, 259.0, 289.5, 320.0, 350.5/

*  BOX BOUNDARIES FOR MONTH_ALL_LEAP
	DATA	(line_mem(i),i=89,101)
     .	/     0.0, 31.0, 60.0, 91.0, 121.0, 152.0, 182.0, 213.0, 
     .      244.0, 274.0, 305.0, 335.0, 366.0/

* Miscellaneous (axis orientations)

	DATA 	axis_orients(  1 )	/ 'WE' /,   ! west to east
     .          axis_orients(  2 )	/ 'SN' /,   ! south to north
     .          axis_orients(  3 )	/ 'DU' /,   ! down to up
     .          axis_orients(  4 )	/ 'TI' /,   ! time (always forward)
     .          axis_orients(  5 )	/ 'EE' /,   ! Ensemble
     .          axis_orients(  6 )	/ 'FI' /,   ! Forecast
     .          axis_orients(  7 )	/ 'UD' /,   ! up to down
     .          axis_orients(  8 )	/ 'NA' /,   ! not applicable
     .          axis_orients(  9 )	/ 'NA' /    ! ... unused slot ...   

	END
