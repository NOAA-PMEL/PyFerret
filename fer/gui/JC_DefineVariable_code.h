/* 
 * JC_DefineVariable_code.h
 *
 * Jonathan Callahan
 * Feb 13th 1996
 *
 * This file contains the auxiliary functions which are included by
 * JC_DefineVariable.c.
 *
 */

/* .................... Function Definitions .................... */

#include <Xm/CascadeB.h>
#include <Xm/MessageB.h>
#include <Xm/SeparatoG.h>
#include "ferret_structures.h"
#include "JC_CallbackUtility.h"
#include "JC_Utility.h"
/*
#include "JC_InterInterface.h"
*/ 

typedef struct {
     char varname[MAX_NAME_LENGTH];
     char function[MAX_NAME_LENGTH];
     char const_1[MAX_NAME_LENGTH];
     char var_1[MAX_NAME_LENGTH];
     char const_2[MAX_NAME_LENGTH];
     char var_2[MAX_NAME_LENGTH];
     char const_3[MAX_NAME_LENGTH];
     char var_3[MAX_NAME_LENGTH];
} JC_Func_Elements;

typedef struct {
     char varname[MAX_NAME_LENGTH];
     char const_1[MAX_NAME_LENGTH];
     char var_1[MAX_NAME_LENGTH];
     char operator[MAX_NAME_LENGTH];
     char const_2[MAX_NAME_LENGTH];
     char var_2[MAX_NAME_LENGTH];
} JC_LinComb_Elements;

typedef struct {
     char varname[MAX_NAME_LENGTH];
     char const_1[MAX_NAME_LENGTH];
     char var_1[MAX_NAME_LENGTH];
     char operator[MAX_NAME_LENGTH];
     char const_2[MAX_NAME_LENGTH];
} JC_Exp_Elements;

typedef struct {
     char title[MAX_NAME_LENGTH];
     char units[MAX_NAME_LENGTH];
     char dset[4][MAX_NAME_LENGTH]; /* for 4 possible dsets needed with the "THETA_F0" function */
} JC_Title_Elements;

JC_DefinedVariable my_test_DV;

extern JC_Variable GLOBAL_Variable;
extern JC_Region GLOBAL_Region;
extern JC_Regridding GLOBAL_Regridding;
extern JC_StateFlags GLOBAL_StateFlags;

extern LIST *GLOBAL_DatasetList;
extern LIST *GLOBAL_GlobalVariableList;

extern void JC_LetCommand_Create( char *command, JC_DefinedVariable *DV_ptr );
extern void JC_ListTraverse_FoundDsetMatch( char *data, char *curr );
extern JC_Object *JC_Clone_ReturnPointer( char *var, char *dset );
extern void JC_DatasetElement_QueryFerret( JC_DatasetElement *this, Boolean new_dataset );

extern Boolean JC_DefineVariable_is_displayed;
extern void JC_II_SelectMenus_Recreate( swidget caller_id );
extern void JC_II_MainMenu_Maintain( JC_StateFlags *SF_ptr );
 
extern void JC_LetCommand_CreateClonedVarExpression(char *var_expression, int var_ID , JC_DefinedVariable *DV_ptr);
extern void JC_Clone_Print( JC_Object *this, FILE *File_ptr );

static JC_Func_Elements F_elems={ NULL, };
static JC_LinComb_Elements LC_elems={ NULL, };
static JC_Exp_Elements EXP_elems={ NULL, };
static JC_Title_Elements Title_elems={ NULL, };

int GLOBAL_func_type=0;

static char let_command[MAX_COMMAND_LENGTH];

static Widget JC_DV_DsetMenu=NULL; 
Widget JC_DV_SelectMenu1_F=NULL;
Widget JC_DV_SelectMenu2_F=NULL;
Widget JC_DV_SelectMenu3_F=NULL;
Widget JC_DV_SelectMenu1_LC=NULL;
Widget JC_DV_SelectMenu2_LC=NULL;
Widget JC_DV_SelectMenu1_EXP=NULL;
static Widget JC_DV_FunctionMenu_F=NULL;
static Widget JC_DV_OperatorMenu_LC=NULL;
static Widget JC_DV_OperatorMenu_EXP=NULL;

static void JC_DV_Initialize( int function_type );
 
/* .................... Internal Declarations .................... */
 
static void JC_DV_OperatorMenu_LC_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_DV_OperatorMenu_EXP_CB( Widget wid, XtPointer client_data, XtPointer call_data );
 
static void JC_RG_Button1_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_RG_Button2_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_RG_Button3_CB( Widget wid, XtPointer client_data, XtPointer call_data );
 
static void JC_DV_DsetMenu_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_DV_FunctionMenu_CB( Widget wid, XtPointer client_data, XtPointer call_data );

void JC_DV_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() );
void JC_DV_SelectMenuButton_F1_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuButton_F2_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuButton_F3_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuButton_LC1_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuButton_LC2_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuButton_EXP1_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuF1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuF2_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuF3_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuLC1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuLC2_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_DV_SelectMenuEXP1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
 
 
/* .................... Menu Items .................... */
 
JC_MenuItem JC_DV_menu_functions[] = {
{ "INT", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "INT", NULL },
{ "MAX", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "MAX", NULL },
{ "MIN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "MIN", NULL },
{ "ABS", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "ABS", NULL },
{ "EXP", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "EXP", NULL },
{ "LN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "LN", NULL },
{ "LOG", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "LOG", NULL },
{ "SIN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "SIN", NULL },
{ "COS", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "COS", NULL },
{ "TAN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "TAN", NULL },
{ "ASIN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "ASIN", NULL },
{ "ACOS", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "ACOS", NULL },
{ "ATAN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "ATAN", NULL },
{ "ATAN2", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "ATAN2", NULL },
{ "MOD", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "MOD", NULL },
{ "MISSING", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "MISSING", NULL },
{ "IGNORE0", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "IGNORE0", NULL },
{ "RANDU", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "RANDU", NULL },
{ "RANDN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "RANDN", NULL },
/*
   { "RHO_UN", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "RHO_UN", NULL },
   { "THETA_FO", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_FunctionMenu_CB, "THETA_FO", NULL },
   */
NULL,
};

JC_MenuItem JC_DV_menu_datasets[] = {
{ NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
{ NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
{ NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
{ NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL }
};

JC_MenuItem JC_DV_menu_operators_EXP[] = {
{ "+", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_EXP_CB, "+", NULL },
{ "-", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_EXP_CB, "-", NULL },
{ "*", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_EXP_CB, "*", NULL },
{ "/", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_EXP_CB, "/", NULL },
{ "^", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_EXP_CB, "^", NULL },
NULL,
};

JC_MenuItem JC_DV_menu_operators_LC[] = {
{ "+", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_LC_CB, "+", NULL },
{ "-", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_LC_CB, "-", NULL },
{ "*", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_LC_CB, "*", NULL },
{ "/", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_LC_CB, "/", NULL },
{ "^", &xmPushButtonGadgetClass, NULL, NULL, NULL, JC_DV_OperatorMenu_LC_CB, "^", NULL },
NULL,
};
