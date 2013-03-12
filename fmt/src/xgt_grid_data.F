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
* names and sizes of predefined grids for program GFDL
* And climatological axes

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

	include 'tmap_dims.parm'
	include 'implicit.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xunits.cmn_text'
	include 'xtm_grid.cmn_text'

	INTEGER		ngrd_def, nlne_def, grids_left, lines_left,
     .			grid_lines_left, i, j,
     .			pline_mem_next, pline_mem_left, outtype

	PARAMETER	(ngrd_def	= 5,
     .			 nlne_def	= 13,
     .			 grids_left	= grid_ceiling-ngrd_def,
     .			 lines_left	= line_ceiling-nlne_def,
     .			 grid_lines_left= 4*grids_left,
     .                   pline_mem_next = 614,
     .                   pline_mem_left = maxlinestore-pline_mem_next+1)

	PARAMETER	(outtype	= 6)  ! netcdf type double, output axis type.

* LAST LINE_MEM POSITION + 1
	DATA	next_line_mem_pos  /pline_mem_next/
        DATA    pline_mem_init / pline_mem_next/

* DEFINE LINES FIRST
	DATA
     .		line_name	(mpsxt)	/ 'PSXT'	/,
     .		line_name_orig	(mpsxt)	/ 'PSXT'	/,
     .		line_subsc1	(mpsxt) / unspecified_int4 /,
     .		line_dim	(mpsxt)	/ 160		/,
     .		line_units	(mpsxt)	/ 'LONGITUDE'	/,
     .		line_unit_code  (mpsxt) / pun_degrees   /,
     .		line_direction	(mpsxt)	/ 'WE'		/,
     .		line_regular	(mpsxt)	/ .TRUE.	/,
     .		line_modulo	(mpsxt) / .TRUE.        /,
     .		line_modulo_len	(mpsxt) / 360.0D0       /,
     .		line_start	(mpsxt)	/ 130.5		/,
     .		line_delta	(mpsxt)	/ 1.0		/,
     .		line_tunit	(mpsxt)	/ real4_init /,
     .		line_t0		(mpsxt)	/ char_init20 /,
     .		line_cal_name	(mpsxt)	/'GREGORIAN'/
	DATA
     .		line_name	(mpsxu)	/ 'PSXU'	/,
     .		line_name_orig	(mpsxu)	/ 'PSXU'	/,
     .		line_subsc1	(mpsxu) / unspecified_int4 /,
     .		line_dim	(mpsxu)	/ 160		/,
     .		line_units	(mpsxu)	/ 'LONGITUDE'	/,
     .		line_unit_code  (mpsxu) / pun_degrees   /,
     .		line_direction	(mpsxu)	/ 'WE'		/,
     .		line_regular	(mpsxu)	/ .TRUE.	/,
     .		line_modulo	(mpsxu) / .TRUE.        /,
     .		line_modulo_len	(mpsxu) / 360.0D0       /,
     .		line_start	(mpsxu)	/ 131.0		/,
     .		line_delta	(mpsxu)	/ 1.0		/,
     .		line_tunit	(mpsxu)	/ real4_init /,
     .		line_t0		(mpsxu)	/ char_init20 /,
     .		line_cal_name	(mpsxu)	/'GREGORIAN'/
	DATA
     .		line_name	(mpsyt)	/ 'PSYT'	/,
     .		line_name_orig	(mpsyt)	/ 'PSYT'	/,
     .		line_subsc1	(mpsyt) / 2		/,
     .		line_dim	(mpsyt)	/ 100		/,
     .		line_units	(mpsyt)	/ 'LATITUDE'	/,
     .		line_unit_code  (mpsyt) / pun_degrees   /,
     .		line_direction	(mpsyt)	/ 'SN'		/,
     .		line_regular	(mpsyt)	/ .FALSE.	/,
     .		line_modulo	(mpsyt) / .FALSE.       /,
     .		line_modulo_len	(mpsyt) / 0.0D0         /,
     .		line_start	(mpsyt)	/ unspecified_val8 /,
     .		line_delta	(mpsyt)	/ unspecified_val8 /,
     .		line_tunit	(mpsyt)	/ real4_init /,
     .		line_t0		(mpsyt)	/ char_init20 /,
     .		line_cal_name	(mpsyt)	/'GREGORIAN'/
	DATA
     .		line_name	(mpsyu)	/ 'PSYU'	/,
     .		line_name_orig	(mpsyu)	/ 'PSYU'	/,
     .		line_subsc1	(mpsyu) / 203		/,
     .		line_dim	(mpsyu)	/ 100		/,
     .		line_units	(mpsyu)	/ 'LATITUDE'	/,
     .		line_unit_code  (mpsyu) / pun_degrees   /,
     .		line_direction	(mpsyu)	/ 'SN'		/,
     .		line_regular	(mpsyu)	/ .FALSE.	/,
     .		line_modulo	(mpsyu) / .FALSE.       /,
     .		line_modulo_len	(mpsyu) / 0.0D0         /,
     .		line_start	(mpsyu)	/ unspecified_val8 /,
     .		line_delta	(mpsyu)	/ unspecified_val8 /,
     .		line_tunit	(mpsyu)	/ real4_init /,
     .		line_t0		(mpsyu)	/ char_init20 /,
     .		line_cal_name	(mpsyu)	/'GREGORIAN'/
	DATA
     .		line_name	(mpszt)	/ 'PSZT'	/,
     .		line_name_orig	(mpszt)	/ 'PSZT'	/,
     .		line_subsc1	(mpszt) / 404		/,
     .		line_dim	(mpszt)	/ 27		/,
     .		line_units	(mpszt)	/ 'METERS'	/,
     .		line_unit_code  (mpszt) / pun_meters   /,
     .		line_direction	(mpszt)	/ 'UD'		/,
     .		line_regular	(mpszt)	/ .FALSE.	/,
     .		line_modulo	(mpszt) / .FALSE.       /,
     .		line_modulo_len	(mpszt) / 0.0D0         /,
     .		line_start	(mpszt)	/ unspecified_val8 /,
     .		line_delta	(mpszt)	/ unspecified_val8 /,
     .		line_tunit	(mpszt)	/ real4_init /,
     .		line_t0		(mpszt)	/ char_init20 /,
     .		line_cal_name	(mpszt)	/'GREGORIAN'/

	DATA
     .		line_name	(mpszw)	/ 'PSZW'	/,
     .		line_name_orig	(mpszw)	/ 'PSZW'	/,
     .		line_subsc1	(mpszw) / 459		/,
     .		line_dim	(mpszw)	/ 27		/,
     .		line_units	(mpszw)	/ 'METERS'	/,
     .		line_unit_code  (mpszw) / pun_meters   /,
     .		line_direction	(mpszw)	/ 'UD'		/,
     .		line_regular	(mpszw)	/ .FALSE.	/,
     .		line_modulo	(mpszw) / .FALSE.       /,
     .		line_modulo_len	(mpszw) / 0.0D0         /,
     .		line_start	(mpszw)	/ unspecified_val8 /,
     .		line_delta	(mpszw)	/ unspecified_val8 /
     .		line_tunit	(mpszw)	/ real4_init /,
     .		line_t0		(mpszw)	/ char_init20 /,
     .		line_cal_name	(mpszw)	/'GREGORIAN'/


	DATA
     .		line_name	(clmirr)	/ 'MONTH_IRREG'	/,
     .		line_name_orig	(clmirr)	/ 'MONTH_IRREG'	/,
     .		line_subsc1	(clmirr)	/ 514		/,
     .		line_dim	(clmirr)	/ 12		/,
     .		line_units	(clmirr)	/ 'DAYS'	/,
     .		line_unit_code  (clmirr)	/ -4   /,
     .		line_direction	(clmirr)	/ 'TI'		/,
     .		line_regular	(clmirr)	/ .FALSE.	/,
     .		line_modulo	(clmirr)	/ .TRUE.       /,
     .		line_modulo_len	(clmirr)	/ 365.2425         /,
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
     .		line_modulo_len	(clmreg)	/ 8765.82         /,
     .		line_start	(clmreg)	/ 366 /,
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
     .		line_modulo_len	(clseas)	/ 8765.82         /,
     .		line_start	(clseas)	/ 1095.7275 /,
     .		line_delta	(clseas)	/ 2191.455 /,
     .		line_tunit	(clseas)	/ 3600 /,
     .		line_t0		(clseas)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clseas)	/'GREGORIAN'/

	DATA
     .		line_name	(clmgrg)	/ 'MONTH_GREGORIAN' /,
     .		line_name_orig	(clmgrg)	/ 'MONTH_GREGORIAN' /,
     .		line_subsc1	(clmgrg)	/ 539		/,
     .		line_dim	(clmgrg)	/ 12		/,
     .		line_units	(clmgrg)	/ 'days'	/,
     .		line_unit_code  (clmgrg)	/ -4   /,
     .		line_direction	(clmgrg)	/ 'TI'		/,
     .		line_regular	(clmgrg)	/ .FALSE.	/,
     .		line_modulo	(clmgrg)	/ .TRUE.       /,
     .		line_modulo_len	(clmgrg)	/ 365.2425         /,
     .		line_start	(clmgrg)	/ unspecified_val8 /,
     .		line_delta	(clmgrg)	/ unspecified_val8 /,
     .		line_tunit	(clmgrg)	/ 86400 /,
     .		line_t0		(clmgrg)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clmgrg)	/'GREGORIAN'/
     
	DATA
     .		line_name	(clmnol)	/ 'MONTH_NOLEAP' /,
     .		line_name_orig	(clmnol)	/ 'MONTH_NOLEAP' /,
     .		line_subsc1	(clmnol)	/ 564		/,
     .		line_dim	(clmnol)	/ 12		/,
     .		line_units	(clmnol)	/ 'days'	/,
     .		line_unit_code  (clmnol)	/ -4   /,
     .		line_direction	(clmnol)	/ 'TI'		/,
     .		line_regular	(clmnol)	/ .FALSE.	/,
     .		line_modulo	(clmnol)	/ .TRUE.       /,
     .		line_modulo_len	(clmnol)	/ 365.         /,
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
     .		line_modulo_len	(clm360)	/ 360.         /,
     .		line_start	(clm360)	/ 15. /,
     .		line_delta	(clm360)	/ 30. /,
     .		line_tunit	(clm360)	/ 86400 /,
     .		line_t0		(clm360)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clm360)	/'360_DAY'/

	DATA
     .		line_name	(clmall)	/ 'MONTH_ALL_LEAP' /,
     .		line_name_orig	(clmall)	/ 'MONTH_ALL_LEAP' /,
     .		line_subsc1	(clmall)	/ 589	/,
     .		line_dim	(clmall)	/ 12		/,
     .		line_units	(clmall)	/ 'days'	/,
     .		line_unit_code  (clmall)	/ -4   /,
     .		line_direction	(clmall)	/ 'TI'		/,
     .		line_regular	(clmall)	/ .FALSE.	/,
     .		line_modulo	(clmall)	/ .TRUE.       /,
     .		line_modulo_len	(clmall)	/ 366.         /,
     .		line_start	(clmall)	/ unspecified_val8 /,
     .		line_delta	(clmall)	/ unspecified_val8 /,
     .		line_tunit	(clmall)	/ 86400 /,
     .		line_t0		(clmall)	/'01-JAN-0000 00:00:00'/,
     .		line_cal_name	(clmall)	/'ALL_LEAP'/


* 7/99 *kob* add in initialization for line_name and mpsnorm.  
*            needed for v3.4 of netcdf use
	DATA    line_name       (mpsnorm) /' '/
	DATA    line_name_orig  (mpsnorm) /' '/

* DEFINE GRIDS NEXT
* 2 dimensional defining grid for wind stress magnitude
	DATA	grid_name  (mps2dt)    / 'PS2DT'	/,
     .		grid_rotation (mps2dt) / 0.0		/,
     .		grid_line  (1, mps2dt) / mpsxt	/,
     .		grid_line  (2, mps2dt) / mpsyt	/,
     .		grid_line  (3, mps2dt) / mpsnorm	/,
     .		grid_line  (4, mps2dt) / mpsunkn	/,
     .		grid_out_prod (1, mps2dt) / .TRUE.   /,
     .		grid_out_prod (2, mps2dt) / .TRUE.   /,
     .		grid_out_prod (3, mps2dt) / .TRUE.   /,
     .		grid_out_prod (4, mps2dt) / .TRUE.   /

* 2 dimensional defining grid for taux and tauy
	DATA	grid_name  (mps2du)    / 'PS2DU'	/,
     .		grid_rotation (mps2du) / 0.0		/,
     .		grid_line  (1, mps2du) / mpsxu	/,
     .		grid_line  (2, mps2du) / mpsyu	/,
     .		grid_line  (3, mps2du) / mpsnorm	/,
     .		grid_line  (4, mps2du) / mpsunkn	/,
     .		grid_out_prod (1, mps2du) / .TRUE.   /,
     .		grid_out_prod (2, mps2du) / .TRUE.   /,
     .		grid_out_prod (3, mps2du) / .TRUE.   /,
     .		grid_out_prod (4, mps2du) / .TRUE.   /

* 3 dimensional defining grid for temp and salt
	DATA	grid_name  (mps3dt)    / 'PS3DT'	/,
     .		grid_rotation (mps3dt) / 0.0		/,
     .		grid_line  (1, mps3dt) / mpsxt	/,
     .		grid_line  (2, mps3dt) / mpsyt	/,
     .		grid_line  (3, mps3dt) / mpszt	/,
     .		grid_line  (4, mps3dt) / mpsunkn	/,
     .		grid_out_prod (1, mps3dt) / .TRUE.   /,
     .		grid_out_prod (2, mps3dt) / .TRUE.   /,
     .		grid_out_prod (3, mps3dt) / .TRUE.   /,
     .		grid_out_prod (4, mps3dt) / .TRUE.   /

* 3 dimensional defining grid for u and v
	DATA	grid_name  (mps3du)    / 'PS3DU'	/,
     .		grid_rotation (mps3du) / 0.0		/,
     .		grid_line  (1, mps3du) / mpsxu	/,
     .		grid_line  (2, mps3du) / mpsyu	/,
     .		grid_line  (3, mps3du) / mpszt	/,
     .		grid_line  (4, mps3du) / mpsunkn	/,
     .		grid_out_prod (1, mps3du) / .TRUE.   /,
     .		grid_out_prod (2, mps3du) / .TRUE.   /,
     .		grid_out_prod (3, mps3du) / .TRUE.   /,
     .		grid_out_prod (4, mps3du) / .TRUE.   /

* 3 dimensional defining grid for w
	DATA	grid_name  (mps3dw)    / 'PS3DW'	/,
     .		grid_rotation (mps3dw) / 0.0		/,
     .		grid_line  (1, mps3dw) / mpsxt	/,
     .		grid_line  (2, mps3dw) / mpsyt	/,
     .		grid_line  (3, mps3dw) / mpszw	/,
     .		grid_line  (4, mps3dw) / mpsunkn	/,
     .		grid_out_prod (1, mps3dw) / .TRUE.   /,
     .		grid_out_prod (2, mps3dw) / .TRUE.   /,
     .		grid_out_prod (3, mps3dw) / .TRUE.   /,
     .		grid_out_prod (4, mps3dw) / .TRUE.   /

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

	DATA	((grid_line(i,j),i=1,4),j=ngrd_def+1,grid_ceiling)
     .			/grid_lines_left*int4_init/

	DATA	(grid_rotation(i),i=ngrd_def+1,grid_ceiling)
     .			/grids_left*real4_init/

	DATA	((grid_out_prod(i,j),i=1,4),j=ngrd_def+1,grid_ceiling)
     .			/grid_lines_left*.FALSE./

* DEFINE LINE AND BOX COORDS NEXT TO LAST
*  COORDS FOR PSYT
	DATA	(line_mem(i),i=2,101)
     .	/-28.83607292175293, -26.52994918823242, -24.28807449340820, 
     .	 -22.15015602111816, -20.15135765075684, -18.32076263427734,
     .	 -16.68010330200195, -15.24281406402588, -14.01343536376953,
     .	 -12.98742485046387, -12.15135097503662, -11.48348140716553,
     .	 -10.95473194122314, -10.52993869781494, -10.16939353942871,
     .	  -9.83332061767578,  -9.49998760223388,  -9.16665363311767,
     .    -8.83331966400146,  -8.49998569488525,  -8.16665267944335,
     .	  -7.83331871032714,  -7.49998474121093,  -7.16665124893188,
     .	  -6.83331823348999,  -6.49998521804809,  -6.16665172576904,
     .	  -5.83331823348999,  -5.49998521804809,  -5.16665172576904,
     .	  -4.83331871032714,  -4.49998521804809,  -4.16665172576904,
     .	  -3.83331871032714,  -3.49998521804809,  -3.16665172576904,
     .	  -2.83331847190856,  -2.49998521804809,  -2.16665196418762,
     .	  -1.83331859111785,  -1.49998521804809,  -1.16665184497833,
     .	  -0.83331835269927,  -0.49998494982719,  -0.16665154695510,
     .	   0.16668185591697,   0.50001525878906,   0.83334869146347,
     .	   1.16668212413787,   1.50001549720764,   1.83334898948669,
     .	   2.16668248176574,   2.50001597404480,   2.83334946632385,
     .	   3.16668295860290,   3.50001621246337,   3.83334970474243,
     .	   4.16668319702148,   4.50001621246337,   4.83334970474243,
     .	   5.16668319702148,   5.50001621246337,   5.83334970474243,
     .	   6.16668272018432,   6.50001621246337,   6.83334970474243,
     .	   7.16668272018432,   7.50001668930053,   7.83335018157959,
     .	   8.16668415069580,   8.50001811981201,   8.83335113525390,
     .	   9.16668510437011,   9.50001907348632,   9.83335304260253,
     .	  10.16793632507324,  10.51373767852783,  10.88928699493408,
     .	  11.31389617919922,  11.80609893798828,  12.38336753845215,
     .	  13.06183147430420,  13.85602283477783,  14.77865123748779,
     .	  15.84039688110352,  17.04974937438965,  18.41287040710449,
     .	  19.93349456787109,  21.61287307739258,  23.44975662231445,
     .	  25.44040679931641,  27.57866477966309,  29.85604095458984,
     .	  32.26185226440430,  34.78339004516602,  37.40612411499023,
     .	  40.11392593383789,  42.88932037353516,  45.71377182006836,
     .	  48.56797027587891/

*  BOX BOUNDARIES FOR PSYT
*   note: Box boundaries on the YT axis DO NOT lie midway between grid points
*	  They are constructed to be consistent with the variables
*	  PSIT and DYT of the Phil/Seigel model
	DATA	(line_mem(i),i=102,202)/
     .	 -30.00000059604645, -27.67214524745941, -25.38775503635406,
     .	 -23.18839609622955, -21.11191737651825, -19.19079816341400,
     .	 -17.45072722434997, -15.90948104858398, -14.57614636421204,
     .	 -13.45072317123413, -12.52412497997284, -11.77857542037964,
     .	 -11.18838566541672, -10.72107663750648, -10.33879896998405,
     .	  -9.99998718500137,  -9.66665378212929,  -9.33332037925720,
     .	  -8.99998697638512,  -8.66665357351303,  -8.33332017064095,
     .	  -7.99998676776886,  -7.66665336489677,  -7.33331996202469,
     .	  -6.99998655915260,  -6.66665315628052,  -6.33331975340843,
     .	  -5.99998635053635,  -5.66665294766426,  -5.33331954479218,
     .	  -4.99998614192009,  -4.66665273904800,  -4.33331933617592,
     .	  -3.99998593330383,  -3.66665253043175,  -3.33331912755966,
     .	  -2.99998572468758,  -2.66665232181549,  -2.33331891894340,
     .	  -1.99998551607132,  -1.66665211319923,  -1.33331871032715,
     .	  -0.99998530745506,  -0.66665190458298,  -0.33331850171089,
     .	   0.00001490116119,   0.33334830403328,   0.66668170690537,
     .	   1.00001510977745,   1.33334851264954,   1.66668191552162,
     .	   2.00001531839371,   2.33334872126579,   2.66668212413788,
     .	   3.00001552700996,   3.33334892988205,   3.66668233275414,
     .	   4.00001573562622,   4.33334913849831,   4.66668254137039,
     .	   5.00001594424248,   5.33334934711456,   5.66668274998665,
     .	   6.00001615285874,   6.33334955573082,   6.66668295860291,
     .	   7.00001636147499,   7.33334976434708,   7.66668316721916,
     .	   8.00001657009125,   8.33334997296333,   8.66668337583542,
     .	   9.00001677870751,   9.33335018157959,   9.66668358445168,
     .	  10.00001698732376,  10.33585005998612,  10.69162029027939,
     .	  11.08694973587990,  11.54083672165871,  12.07135620713234,
     .	  12.69537439942360,  13.42828330397606,  14.28375723958016,
     .	  15.27353766560555,  16.40724852681160,  17.69224312901497,
     .	  19.13348940014839,  20.73349097371102,  22.49224796891213,
     .	  24.40725651383400,  26.47354879975319,  28.68377104401588,
     .	  31.02829995751381,  33.49539366364479,  36.07137742638588,
     .	  38.74085965752602,  41.48697438836098,  44.29164591431618,
     .	  47.13587656617165,  50.00004377961159/

*  COORDS FOR PSYU
	DATA	(line_mem(i),i=203,302)
     .	/-27.67214393615723, -25.38775444030762, -23.18839454650879,
     .	 -21.11191749572754, -19.19079780578613, -17.45072746276855,
     .	 -15.90948104858398, -14.57614612579346, -13.45072364807129,
     .	 -12.52412509918213, -11.77857589721680, -11.18838596343994,
     .	 -10.72107696533203, -10.33879947662354,  -9.99998760223388,
     .	  -9.66665458679199,  -9.33332061767578,  -8.99998664855957,
     .	  -8.66665267944335,  -8.33331966400146,  -7.99998569488525,
     .	  -7.66665172576904,  -7.33331823348999,  -6.99998474121093,
     .	  -6.66665124893188,  -6.33331823348999,  -5.99998474121093,
     .	  -5.66665172576904,  -5.33331823348999,  -4.99998474121093,
     .	  -4.66665172576904,  -4.33331823348999,  -3.99998497962951,
     .	  -3.66665172576904,  -3.33331847190856,  -2.99998521804809,
     .	  -2.66665196418762,  -2.33331847190856,  -1.99998533725738,
     .	  -1.66665184497833,  -1.33331847190856,  -0.99998503923416,
     .	  -0.66665166616439,  -0.33331826329231,   0.00001515448275,
     .	   0.33334857225418,   0.66668194532394,   1.00001537799835,
     .	   1.33334875106811,   1.66668212413787,   2.00001549720764,
     .	   2.33334898948669,   2.66668272018432,   3.00001621246337,
     .	   3.33334970474243,   3.66668295860290,   4.00001621246337,
     .	   4.33334970474243,   4.66668272018432,   5.00001621246337,
     .	   5.33334922790527,   5.66668272018432,   6.00001621246337,
     .	   6.33334922790527,   6.66668272018432,   7.00001573562622,
     .	   7.33334970474243,   7.66668319702148,   8.00001716613769,
     .	   8.33335113525390,   8.66668415069580,   9.00001811981201,
     .	   9.33335208892822,   9.66668605804443,  10.00001907348633,
     .	  10.33585262298584,  10.69162178039551,  11.08695220947266,
     .	  11.54083919525146,  12.07135868072510,  12.69537734985352,
     .	  13.42828655242920,  14.28376007080078,  15.27354145050049,
     .	  16.40725135803223,  17.69224548339844,  19.13349342346191,
     .	  20.73349571228027,  22.49225234985352,  24.40726089477539,
     .	  26.47355461120605,  28.68377685546875,  31.02830314636230,
     .	  33.49539947509766,  36.07138442993164,  38.74086761474609,
     .	  41.48698425292969,  44.29165267944336,  47.13588333129883,
     .	  50.00005340576172/

*  BOX BOUNDARIES FOR PSYU
*   note: Box boundaries on the YU axis DO lie midway between grid points
	DATA	(line_mem(i),i=303,403)/
     .	 -28.83607161045075, -26.52994883060456, -24.28807413578034,
     .	 -22.15015542507172, -20.15135645866394, -18.32076144218445,
     .	 -16.68010282516480, -15.24281239509583, -14.01343345642090,
     .	 -12.98742270469666, -12.15134876966477, -11.48347908258438,
     .	 -10.95472967624665, -10.52993631362915, -10.16939160227776,
     .	  -9.83331900835038,  -9.49998560547829,  -9.16665220260620,
     .	  -8.83331879973412,  -8.49998539686203,  -8.16665199398995,
     .	  -7.83331859111786,  -7.49998518824578,  -7.16665178537369,
     .	  -6.83331838250161,  -6.49998497962952,  -6.16665157675743,
     .	  -5.83331817388535,  -5.49998477101326,  -5.16665136814118,
     .	  -4.83331796526909,  -4.49998456239701,  -4.16665115952492,
     .	  -3.83331775665284,  -3.49998435378075,  -3.16665095090866,
     .	  -2.83331754803658,  -2.49998414516449,  -2.16665074229241,
     .	  -1.83331733942032,  -1.49998393654824,  -1.16665053367615,
     .	  -0.83331713080407,  -0.49998372793198,  -0.16665032505989,
     .	   0.16668307781219,   0.50001648068428,   0.83334988355636,
     .	   1.16668328642845,   1.50001668930053,   1.83335009217262,
     .	   2.16668349504470,   2.50001689791679,   2.83335030078888,
     .	   3.16668370366096,   3.50001710653305,   3.83335050940513,
     .	   4.16668391227722,   4.50001731514930,   4.83335071802139,
     .	   5.16668412089348,   5.50001752376556,   5.83335092663765,
     .	   6.16668432950973,   6.50001773238182,   6.83335113525390,
     .	   7.16668453812599,   7.50001794099807,   7.83335134387016,
     .	   8.16668474674225,   8.50001814961433,   8.83335155248642,
     .	   9.16668495535850,   9.50001835823059,   9.83335176110267,
     .	  10.16793498396873,  10.51373663544655,  10.88928645849228,
     .	  11.31389468908310,  11.80609792470932,  12.38336676359176,
     .	  13.06183034181595,  13.85602176189422,  14.77864891290664,
     .	  15.84039455652237,  17.04974728822708,  18.41286772489547,
     .	  19.93349176645279,  21.61287111043930,  23.44975394010544,
     .	  25.44040447473526,  27.57866197824478,  29.85603767633438,
     .	  32.26184898614883,  34.78338772058487,  37.40612083673477,
     .	  40.11391931772232,  42.88931232690811,  45.71376329660415,
     .	  48.56796222925186,  51.43212944269180/

*  COORDS FOR PSZT
	DATA	(line_mem(i),i=404,430)
     .	/    5.00,   15.00,   25.00,   35.00,   45.00,   55.00,   65.00,
     .	    75.00,   85.00,   95.00,  106.25,  120.00,  136.25,  155.00,
     .	   177.50,  205.00,  240.00,  288.50,  362.50,  483.50,  680.00,
     .	   979.50, 1395.50, 1916.00, 2524.00, 3174.00, 3824.00/

*  BOX BOUNDARIES FOR PSZT
	DATA	(line_mem(i),i=431,458)
     .	/    0.00,   10.00,   20.00,   30.00,   40.00,   50.00,   60.00,
     .	    70.00,   80.00,   90.00,  100.00,  112.50,  127.50,  145.00,
     .	   165.00,  190.00,  220.00,  260.00,  317.00,  408.00,  559.00,
     .	   801.00, 1158.00, 1633.00, 2199.00, 2849.00, 3499.00, 4149.00/

*  COORDS FOR PSZW
	DATA	(line_mem(i),i=459,485)
     .	/   10.00,   20.00,   30.00,   40.00,   50.00,   60.00,   70.00,
     .	    80.00,   90.00,  100.00,  112.50,  127.50,  145.00,  165.00,
     .	   190.00,  220.00,  260.00,  317.00,  408.00,  559.00,  801.00,
     .	  1158.00, 1633.00, 2199.00, 2849.00, 3499.00, 4149.00/

*  BOX BOUNDARIES FOR PSZW
	DATA	(line_mem(i),i=486,513)
     .	/    5.00,   15.00,   25.00,   35.00,   45.00,   55.00,   65.00,
     .	    75.00,   85.00,   95.00,  106.25,  120.00,  136.25,  155.00,
     .	   177.50,  205.00,  240.00,  288.50,  362.50,  483.50,  680.00,
     .	   979.50, 1395.50, 1916.00, 2524.00, 3174.00, 3824.00, 4149.00/

* MONTH_IRREG

*  COORDS FOR MONTH_IRREG
	DATA	(line_mem(i),i=514,525)
     .	/     15.5, 45.1, 74.7, 105.2, 135.7, 166.2, 196.7, 227.7, 
     .	     258.2, 288.7, 319.2,  349.7/

*  BOX BOUNDARIES FOR MONTH_IRREG
	DATA	(line_mem(i),i=526,538)
     .	/     0.0, 31.0, 59.2425, 90.2425, 120.2425, 151.2425, 
     .	    181.2425, 212.2425, 243.2425, 273.2425, 304.2425, 
     .	    334.2425, 365.2425/
     
* MONTH_GREGORIAN
     
*  COORDS FOR MONTH_GREGORIAN
	DATA	(line_mem(i),i=539,550)
     .	/     15.5, 45.1, 74.7, 105.2, 135.7, 166.2, 196.7, 227.7, 
     .	     258.2, 288.7, 319.2,  349.7/

*  BOX BOUNDARIES FOR MONTH_GREGORIAN
	DATA	(line_mem(i),i=551,563)
     .	/     0.0, 31.0, 59.2425, 90.2425, 120.2425, 151.2425, 
     .	    181.2425, 212.2425, 243.2425, 273.2425, 304.2425, 
     .	    334.2425, 365.2425/

* MONTH_NOLEAP
     
*  COORDS FOR MONTH_NOLEAP
	DATA	(line_mem(i),i=564,575)
     .	/     15.5,  45.0,  74.5, 105.0, 135.5, 166.0, 196.5, 
     .	     227.5, 258.0, 288.5, 319.0, 349.5/

*  BOX BOUNDARIES FOR MONTH_NOLEAP
	DATA	(line_mem(i),i=576,588)
     .	/     0.0, 31.0, 59.0, 90.0, 120.0, 151.0, 181.0, 
     .      212.0, 243.0, 273.0, 304.0, 334.0, 365.0/

* MONTH_ALL_LEAP
     
*  COORDS FOR MONTH_ALL_LEAP
	DATA	(line_mem(i),i=589,600)
     .	/     15.5, 45.5, 75.5, 106.0, 136.5, 167.0, 197.5, 
     .       228.5, 259.0, 289.5, 320.0, 350.5/

*  BOX BOUNDARIES FOR MONTH_ALL_LEAP
	DATA	(line_mem(i),i=601,613)
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
