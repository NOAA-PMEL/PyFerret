/* 
 * JC_EditDefinedVar_code.h
 *
 * Jonathan Callahan
 * Jul 12th 1996
 *
 * This file contains the auxiliary functions which are included by
 * JC_EditDefinedVar.c.
 *
 */

/* .................... Function Definitions .................... */

#include "ferret_structures.h"
#include "ferret_shared_buffer.h"
#include "JC_CallbackUtility.h"
#include "JC_Utility.h"
/*
#include "JC_InterInterface.h"
*/ 

/* .................... External Declarations .................... */
 
extern JC_StateFlags GLOBAL_StateFlags;

extern LIST *GLOBAL_DatasetList;

extern char *CollectToReturn(char *targetStr, char *subStr);

extern void JC_ListTraverse_FoundDsetMatch( char *data, char *curr );
extern void JC_DatasetElement_QueryFerret( JC_DatasetElement *this, Boolean new_dataset );

extern Boolean JC_EditDefinedVar_is_displayed;
extern void JC_II_SelectMenus_Recreate( swidget caller_id );
extern void JC_II_MainMenu_Maintain( JC_StateFlags *SF_ptr );
 

/* .................... Internal Declarations .................... */
 
Widget EDV_SelectMenu_widget=NULL;
char assigned_dset_name[MAX_NAME_LENGTH]="";


static void JC_EDV_Cancel_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_EDV_Initialize( void );
static void EDV_Define_CB( Widget wid, XtPointer client_data, XtPointer call_data );

void EDV_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)()  );
void EDV_SelectMenuDefineButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
