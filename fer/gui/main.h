/* main.h
 *
 * John Osborne
 * Jonathan Callahan (after Oct 95)
 *
 */

#include <stdio.h>
#include <sys/types.h>       /* for the stat() call in InitPixmaps() */
#include <sys/stat.h>        /* for the stat() call in InitPixmaps() */
#include <Xm/CascadeB.h>
#include <X11/cursorfont.h>


#include "ferret_fortran.h"
#include "ferret_shared_buffer.h"
#include "ferret_structures.h"

#define DONT_UPDATE_MM 3
/*
#include "map_pn_final.xpm"
*/
#include "v1_op_num.xpm"
#include "v1_op_v2.xpm"
#include "func_v1.xpm"

#include "ave_t.xpm"
#include "ave_x.xpm"
#include "ave_y.xpm"
#include "ave_z.xpm"

#include "ave.xpm"
#include "var.xpm"
#include "sum_x.xpm"
#include "sum_y.xpm"
#include "sum_t.xpm"
#include "sum_z.xpm"
#include "rsum_t.xpm"
#include "rsum_x.xpm"
#include "rsum_y.xpm"
#include "rsum_z.xpm"
#include "shift_t.xpm"
#include "shift_y.xpm"
#include "shift_x.xpm"
#include "shift_z.xpm"
#include "dx_bac.xpm"
#include "dx_ctr.xpm"
#include "dx_for.xpm"
#include "dy_bac.xpm"
#include "dy_ctr.xpm"
#include "dy_for.xpm"
#include "dz_ctr.xpm"
#include "dz_for.xpm"
#include "dz_bac.xpm"
#include "dt_ctr.xpm"
#include "dt_for.xpm"
#include "dt_bac.xpm"
#include "int_def_x.xpm"
#include "int_def_y.xpm"
#include "int_def_z.xpm"
#include "int_def_t.xpm"
#include "int_x.xpm"
#include "int_y.xpm"
#include "int_z.xpm"
#include "int_t.xpm"
#include "boxcar.xpm"
#include "binomial.xpm"
#include "hanning.xpm"
#include "welch.xpm"
#include "parzen.xpm"
#include "ave_linear.xpm"
#include "ave_filled.xpm"
#include "nearest.xpm"
#include "weq.xpm"


#include "JC_OOP.h"
#include "JC_Map.h"
#include "JC_Utility.h"

/* routines from JC_InterInterface.h */
extern void JC_II_MainMenu_Maintain( JC_StateFlags *SF_ptr );

/* create interface prototypes */
extern swidget create_DefineGrid(swidget UxParent);
extern swidget create_CustomLevels(swidget UxParent);
extern swidget create_CommandLine(swidget UxParent);
extern swidget create_MacroManager(swidget UxParent);
extern swidget create_Open_Save_dset(swidget UxParent);
extern swidget create_Viewports(swidget UxParent);
extern swidget create_CommandHelp(swidget UxParent);
extern swidget create_Splash(swidget UxParent);
extern swidget create_ListManager(swidget UxParent);
extern swidget create_ErrorLog(swidget UxParent);
extern swidget create_PrintSetup(swidget UxParent);
extern swidget create_OpenGO(swidget UxParent);
extern swidget create_OpenFile(swidget UxParent);
extern swidget create_PlotOptions(swidget UxParent);
extern swidget create_SaveDataObject(swidget UxParent);
extern swidget create_JC_DefineVariable(swidget UxParent, int function_type);
extern swidget create_JC_SelectRegridding(swidget UxParent);
extern swidget create_JC_EditDefinedVar(swidget UxParent);

/* utility functions */
extern char *CollectToReturn(char *targetStr, char *subStr);
extern char *FormatFloatStr(double inNum);
extern char *LonToFancyLabel(double inLon);
extern char *LatToFancyLabel(double inLon);
extern void AllCaps(Widget wid, XtPointer client_data,
	       XtPointer cbs);
extern void TimeToFancyDate(double *val, char *outDate);
extern double AbsVal(double val);
extern double DateToSecs(char *inDate, int hasYear);


/* MM Functions */
extern char *macroBuffer;
extern void SetRecordBtn(void);
extern void SetStopBtn(void);
extern int AskUser2(Widget parent, char *question, char *ans1, char *ans2, int default_ans);

/* pixmap functions */
Pixmap GetPixmapFromData(char **inData);
Pixmap GetPixmapFromFile(char *inFile);

extern void JC_PlotCommand_Create(char *plot_command, JC_Object *O_ptr, JC_PlotOptions *PO_ptr);

/* main interface functions */

void JC_VariableTextField_Verify_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_XYZTTextField_Verify_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_CloseDataset_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_SaveButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_EditDefinedVar_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DefineGrid_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_Viewports_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_WindowButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_X_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_Y_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_Z_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_T_SpanButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_X_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_Y_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_Z_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_T_AxisButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );

void JC_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );

void JC_ContextFrame_Initialize( void );

void JC_DatasetNameList_Initialize( void );

void JC_GeometryInterfaceLine_NewSpan( JC_Span *S_ptr );

void JC_FixedToggle_CB( Widget wid, XtPointer client_data, XtPointer call_data );

void JC_GeometryMenu_CB( XtPointer UxClientData );
void JC_GeometryMenu_NewGeometry( int geometry );
void JC_GeometryMenu_NewVariable( JC_Variable *V_ptr, int current_geometry );

void JC_MainInterface_NewVariable( char *var_name, char *dset_name, int variable_info );

void JC_MainMenu_SolidLand_CB( void );

void JC_MapShowHide_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_PlotOptions_CB( Widget wid, XtPointer client_data, XtPointer call_data );

void JC_OverlayButton_CB( void );

void JC_PlotButton_CB( void );

void JC_PlotFrame_MaintainButtons( int geometry, int plot_type, JC_StateFlags *SF_ptr );
int  JC_PlotFrame_MaintainRadios( int geometry, int it_is_a_vector, JC_StateFlags *SF_ptr );

void JC_PlotTypeToggle_CB( int plot_type );

void JC_main_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() );
void JC_Main_SelectMenuButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_Main_SelectMenu_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );

void JC_TransArg_CB( Widget widget_id, XtPointer clientData );
void JC_TransMenu_CB( char *trans );
void JC_Transforms_NewRegion( JC_Region *R_ptr );

void JC_scrollBar_CB( XtPointer clientData, XtPointer cbArg );
void JC_scrollBar_NewAxis( int lo_hi_pt, JC_Axis *A_ptr, int value ); 
void JC_scrollBar_NewSpan( int lo_hi_pt, int is_displayed, JC_Span *S_ptr );
void JC_scrollBar_SetValue(Widget scroll, int value);

void JC_sswwMenu_CB( int xyzt, int selection );
void JC_sswwMenu_NewAxis( JC_Axis *A_ptr, Boolean by_index_in_GUI );

void JC_textField_SetValue( Widget textField_widget, int lo_hi_pt, JC_Span *S_ptr );
void JC_textField_NewSpan( int lo_hi_pt, int is_displayed, JC_Span *S_ptr );

void JC_String_CreateFancyLabel( char *string, double value, JC_Span *S_ptr, JC_StateFlags *SF_ptr );


/* JC_OOP internal functions */

void JC_List_AddVectorPairs( LIST *this, char *dset );
int JC_ListTraverse_Sort( char *data, char *curr );
int JC_ListTraverse_fprintf( char *data, char *curr );
int JC_ListTraverse_Dsetfprintf( char *data, char *curr );
int JC_ListTraverse_Dvarfprintf( char *data, char *curr );
int JC_ListTraverse_Cvarfprintf( char *data, char *curr );
int JC_ListTraverse_strstr( char *data, char *curr );
int JC_ListTraverse_FoundMatch( char *data, char *curr );
int JC_ListTraverse_FoundDvarMatch( char *data, char *curr );
int JC_ListTraverse_FoundCvarMatch( char *data, char *curr );
int JC_ListTraverse_FoundDsetMatch( char *data, char *curr );
int JC_ListTraverse_FreeDataset( char *data, char *curr );



void InitFerretStructs(void);
void DoQuit(void);
void MaintainMainMenu(void);
Display *GetCurrDisplay(void);
static void DisableIndPlotRadios(void);
void SetInitialState(void);
void CancelInitialState(void);

/* printing functions */
extern void PrintCmdCB(void);
extern void InitPS(void);

/* command functions */
extern void ferret_command(char *cmdText, int cmdMode);
extern int ferret_query(int query, smPtr sBuffer, char *tag,
		 char *arg1, char *arg2, char *arg3, char *arg4 );


/* plot options */
extern void SetInitialPOState(void);

/* globals */

/* ... JC_ADDITION ... (11/26/95) */

JC_Variable GLOBAL_Variable;
JC_Regridding GLOBAL_Regridding;
JC_Region GLOBAL_Region;
JC_PlotOptions GLOBAL_PlotOptions;
JC_StateFlags GLOBAL_StateFlags;

LIST *GLOBAL_DatasetList;
LIST *GLOBAL_DatasetNameList;
LIST *GLOBAL_GlobalVariableList;
LIST *GLOBAL_GridList;
LIST *GLOBAL_ViewportList;
LIST *GLOBAL_WindowList;

LIST *GLOBAL_PlottedDataList;


int g_num_user_objects, g_num_plot_objects;
int tool_type;

static Widget scrollBar_widget[5][3];
static Widget textField_widget[5][3];

/* ... END JC_ADDITIONS ... */

int gCycleState = 0;
int gMacroIsRecording = 1;
int gMMIsOpen = 0;
int gSomethingIsPlotted = 0;
Boolean gHiRez = False;
Boolean gFerretIsStarting = True;
int gViewportActive=0, gViewportIsCycling=0, gCurrViewportType=-1, 
	gNumViewportCycles[3]={3,1,1}, gCurrViewportCycle=0;

extern swidget Open_Save_dset, ListManager, ErrorLog, PrintSetup,
OpenGO, OpenFile, PlotOptions, SaveDataObject,
JC_SelectRegridding, JC_DefineVariable;

extern char init_command[128];
Widget indAxisButtons[50], varButtons[50];
int numVarButtons = 0;
static Boolean callUpdate;
extern Boolean gMetaCreationActive;
int gStartYear=1900, gEndYear;

/* prototypes to manage the cx interface parts */
void InitGlobalWidgets(void );
static void InitPixmaps(void);
void CloseWinCB(Widget wid, XtPointer clientData, XtPointer callData);

/* globals */
Widget dataSetMenus=NULL;		/* menus with an arbitrary # of  data set buttons */
double axisVectors[5][1500];
int numValsOnAxis[5];
int axesIndices[5][3];			/* lo, pt, hi */
int actualIndices[5][1500];
static enum {xy, xl, yl, pt, no} mapMode, toolMode, dragMode, oldToolMode;
enum {tAxisIsCalendar, tAxisIsDerivedCalendar, tAxisIsClimatology, tAxisIsRaw, tAxisIsIndex} tAxisState;

int geom_axes[16][4] =
	{{0,0,0,0},    {1,0,0,0},    {0,1,0,0},    {0,0,1,0},
  	 {0,0,0,1},    {1,1,0,0},    {1,0,1,0},    {1,0,0,1},
  	 {0,1,1,0},    {0,1,0,1},    {0,0,1,1},    {1,1,1,0},
  	 {1,1,0,1},    {1,0,1,1},    {0,1,1,1},    {1,1,1,1}};
int geom_desirability[9] = {5,6,7,8,1,2,3,4,0};

#define NUM_TRANSFORMS 25

char *transformCodes[] = {"NON", "AVE", "VAR", "SUM", "RSU", "SHF", "MIN", 
				   "MAX", "DDC", "DDF", "DDB", "DIN", "IIN",
				   "SBX", "SBN", "SWL", "SHN", "SPZ", "FAV",
				   "FLN", "FNR", "NGD", "NBD", "LOC", "WEQ"};


/* Old things which should be deleted when they are no longer referenced */

Widget axssww[5],
	cxByWW[5], cxBySS[5];
Widget axtrans[5], axarg[5], axtrnButton[5][30];
Widget lowScroll[5], hiScroll[5], ptScroll[5], geomOptPBs[16];
Widget toggleButton_fixed[4];
