/* ferret_structures.h
 *
 * John Osborne
 * Jonathan Callahan (after Oct. 1995)
 *
 * This header contains all of the defines and typedefs need by the GUI.
 *
 */

#ifndef _FERRET_STRUCTURES_H 
#define _FERRET_STRUCTURES_H


/* .................... Includes .................... */

#include <Xm/Xm.h> /* for the 'Boolean' type */

#include <list.h>  /* locally added list library */
/*
 * I need to outsmart the interpreter in UIMX
 */
/*#include "/home/r3/tmap/local/sun/include/list.h"*/  /* locally added list library */

/* .................... Defines .................... */

#define INTERNAL_ERROR -9999
#define UNSET_VALUE -99
#define IRRELEVANT_AXIS -111

#define UPDATE_COMMAND_WIDGET 1
#define IGNORE_COMMAND_WIDGET 2


/* *kob* 3/25/99 increase MAX_NAME_LENGTH from 64 to 256 */
#define MAX_NAME_LENGTH 256
#define MAX_COMMAND_LENGTH 256
#define MAX_AXIS_SIZE 2048

#define FALSE 0
#define TRUE 1

#define LO 0
#define HI 1
#define PT 2
#define INC 2
#define DELTA 2
#define NDIG 3
#define INDEX 3


/* .................... Enums .................... */

enum { X_AXIS, Y_AXIS, Z_AXIS, T_AXIS, ALL_AXES }
AXIS_type;

enum { PLOT_NONE, PLOT_LINE, PLOT_SCATTER, PLOT_SHADE, PLOT_CONTOUR,
	    PLOT_FILL, PLOT_VECTOR }
PLOT_type;

enum { GEOM_POINT, GEOM_X, GEOM_Y, GEOM_Z, GEOM_T, GEOM_XY, GEOM_XZ,
	    GEOM_XT, GEOM_YZ, GEOM_YT, GEOM_ZT, GEOM_XYZ, GEOM_XYT,
	    GEOM_XZT, GEOM_YZT, GEOM_XYZT }
GEOM_type;

enum { TRANS_NON, TRANS_AVE, TRANS_VAR, TRANS_SUM, TRANS_RSU, TRANS_SHF,
	    TRANS_MIN, TRANS_MAX, TRANS_DDC, TRANS_DDF, TRANS_DDB,
	    TRANS_DIN, TRANS_IIN, TRANS_SBX, TRANS_SBN, TRANS_SWL,
	    TRANS_SHN, TRANS_SPZ, TRANS_FAV, TRANS_FLN, TRANS_FNR,
	    TRANS_NGD, TRANS_NBD, TRANS_LOC, TRANS_WEQ }
TRANS_type;

enum { INDEX_TIME, MODEL_TIME, CALENDAR_TIME, CLIMATOLOGY_TIME }
TIME_type;

enum { FUNC_FUNCTION1, FUNC_FUNCTION2, FUNC_FUNCTION3, FUNC_FUNCTION4, 
	    FUNC_LINEAR_COMBINATION, FUNC_PLUS_CONSTANT, FUNC_EXPONENT }
FUNC_type;

enum { FUNCFUNC_MAX, FUNCFUNC_MIN, FUNCFUNC_INT, FUNCFUNC_ABS, FUNCFUNC_EXP,
	FUNCFUNC_LN, FUNCFUNC_LOG, FUNCFUNC_SIN, FUNCFUNC_COS, FUNCFUNC_TAN, 
	FUNCFUNC_ASIN, FUNCFUNC_ACOS, FUNCFUNC_ATAN, FUNCFUNC_ATAN2, FUNCFUNC_MOD, 
	FUNCFUNC_MISSING, FUNCFUNC_IGNORE0, FUNCFUNC_RANDU, FUNCFUNC_RANDN,
	FUNCFUNC_RHO_UN, FUNCFUNC_THETA_FO }
FUNCFUNC_type;

enum { FUNCOP_ADD, FUNCOP_SUB, FUNCOP_MULT, FUNCOP_DIV, FUNCOP_EXP }
FUNCOP_type;

enum { QUERY_STATUS=1, QUERY_MESSAGE, QUERY_DSET, QUERY_VARIABLE, QUERY_GRID, 
	QUERY_AXIS, QUERY_COORDS, QUERY_TRANS, QUERY_TRNARG, QUERY_LVARS, 
	QUERY_DVARS, QUERY_VBACK, QUERY_DBACK, QUERY_WINDOWS, QUERY_WCURRENT,
	QUERY_VPORTS, QUERY_VCURRENT, QUERY_VIEW, QUERY_DCURRENT, QUERY_SPAN,
	QUERY_GAXIS, QUERY_GCOORDS, QUERY_SSPOINT, QUERY_WWPOINT, QUERY_UVAR_DEFINITION }
QUERY_type;

enum { FWARN_ERROR=1, FWARN_INFO, FWARN_WARNING, FWARN_MSG }
FWARN_type;


/* .................... Typedefs .................... */

typedef struct {
  char name[MAX_NAME_LENGTH];
  char title[MAX_NAME_LENGTH];
  char units[MAX_NAME_LENGTH];
  float value[MAX_AXIS_SIZE];
  float start, delta, end;
  double secsAtT0, secsPerUnit;
  float ww[3];
  int ss[3];
  int xyzt, unit_code, num_points, time_type;
  Boolean is_modulo, has_fancy_labeling, has_regular_spacing;
  Boolean is_expandable;
} JC_Axis;

typedef struct {
  char name[MAX_NAME_LENGTH];
  JC_Axis axis[8];
} JC_Grid;

typedef struct {
  char name[MAX_NAME_LENGTH];
  float arg;
  int code;
  Boolean exists, compresses, accepts_an_argument;
} JC_Transform;

typedef struct {
  char var[5][MAX_NAME_LENGTH];
  char dset[5][MAX_NAME_LENGTH];
  char rg_transform[5][MAX_NAME_LENGTH];
  enum { UNIFORM, NON_UNIFORM } type;
  Boolean is_on;
} JC_Regridding;

typedef struct {
  char title[MAX_NAME_LENGTH];
  double secsAtT0, secsPerUnit;
  float ww[3];
  int ss[3];
  int xyzt, num_points, time_type;
  Boolean is_modulo, has_fancy_labeling;
  Boolean is_expandable;
  Boolean needs_lo_hi_displayed_in_GUI, is_compressed_in_GUI, by_index_in_GUI;
} JC_Span;

typedef struct {
  char name[MAX_NAME_LENGTH];
  JC_Span span[4];
  JC_Transform transform[4];
  int geometry;
} JC_Region;

typedef struct {
  char name[MAX_NAME_LENGTH];
  char title[MAX_NAME_LENGTH];
  char dset[MAX_NAME_LENGTH];
  char grid[MAX_NAME_LENGTH];
  JC_Axis axis[4];
  int okGeoms[16];
} JC_Variable;

typedef struct {
  char name[MAX_NAME_LENGTH];
  JC_Variable variable;
  JC_Regridding regridding;
  JC_Region region;
  Boolean fixed_axis[4], fixed_regridding, is_a_clone;
} JC_Object;

typedef struct {
  JC_Object *clone_ptr[4];
  char generated_definition[MAX_NAME_LENGTH];
  char name[MAX_NAME_LENGTH];
  char assigned_dset[MAX_NAME_LENGTH];
  char multiplier[4][MAX_NAME_LENGTH];
  char var[4][MAX_NAME_LENGTH];
  char dset[4][MAX_NAME_LENGTH];
  char title[MAX_NAME_LENGTH];
  char units[MAX_NAME_LENGTH];
  char operator[MAX_NAME_LENGTH], function[MAX_NAME_LENGTH];
  int type, number_of_vars;
  Boolean defined_by_GUI;
} JC_DefinedVariable;

typedef struct {
	JC_Object *current_clone_ptr;
     int open_datasets;
     int geometry_last_plotted, plot_type_last_plotted;
     Boolean a_plot_exists, a_clone_is_selected, time_resolution_includes_hours;
} JC_StateFlags;


/*-------------------- Now for the Plot Options --------------------*/

typedef struct {
     int automatic;
     int style;
     int symbol;
} JC_OneDOptions;

typedef struct {
     char palette[MAX_NAME_LENGTH];
     float levels[4];
     enum { NO_LEVELS, LAST_LEVELS, NEW_LEVELS } level_type;
     Boolean key, line, no_key;
} JC_TwoDOptions;

typedef struct {
     int aspect;
     int length;
     enum { NO_LENGTH, LAST_LENGTH, NEW_LENGTH } length_type;
     int xskip, yskip;
} JC_VectorOptions;

typedef struct {
     char name[MAX_NAME_LENGTH];
     char title[MAX_NAME_LENGTH];
     JC_OneDOptions oneD_options;
     JC_TwoDOptions twoD_options;
     JC_VectorOptions vector_options;
     float xlimits_ww[3];
     float ylimits_ww[3];
     int xlimits_ss[3];
     int ylimits_ss[3];
     int plot_type;
     Boolean nolabels, setup, transpose, overlay;
} JC_PlotOptions;

typedef struct _JC_PlottedData {
     JC_Object object;
     JC_PlotOptions plot_options;
     struct _JC_PlottedData *underlay;
} JC_PlottedData;

/*-------------------- Typedef for the dataset synchronization list --------------------*/

typedef struct {
	char name[MAX_NAME_LENGTH];
	LIST *varList;
	LIST *dvarList;
	LIST *cvarList;
} JC_DatasetElement;



#endif /* _FERRET_STRUCTURES_H */
