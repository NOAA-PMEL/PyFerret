/* 
 * JC_SelectRegridding_code.h
 *
 * Jonathan Callahan
 * Mar 6th 1996
 *
 * This file contains the auxiliary functions which are included by
 * JC_SelectRegridding.c.
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

enum { RG_LIN, RG_AVE, RG_ASN, RG_VAR, RG_NGD, RG_SUM, RG_MIN, RG_MAX } RG_Transform_type;

extern JC_Variable GLOBAL_Variable;
extern JC_Region GLOBAL_Region;
extern JC_Regridding GLOBAL_Regridding;
extern JC_StateFlags GLOBAL_StateFlags;

extern Boolean JC_SelectRegridding_is_displayed;
extern Boolean JC_SelectRegridding_is_uniform;
extern void JC_II_Synchronize( swidget caller_id );
extern JC_Regridding_Initialize( JC_Regridding *RG_ptr );
extern void JC_II_FixRegridding( swidget caller_id );
extern void JC_II_ChangeRegriddingLabel( swidget caller_id );

Widget SelectMenu_widget[5]={ NULL, NULL, NULL, NULL, NULL };
static Widget textField_widget[5]={ NULL, NULL, NULL, NULL, NULL };
static Widget TransMenu_widget[5]={ NULL, NULL, NULL, NULL, NULL };
static Widget rowColumn_widget[5]={ NULL, NULL, NULL, NULL, NULL };
static Widget TransMenuButton_widget[5][8];

/* .................... Internal Declarations .................... */

void JC_SR_Initialize( void );
static void JC_SR_TransMenu_Initialize( JC_Regridding *RG_ptr, int xyzt );
static void JC_NonUniform_Setup( JC_Region *R_ptr, JC_Regridding *RG_ptr );
static void JC_RegriddingWidgets_Init( void );

static void JC_SR_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() );
void JC_SR_SelectMenuButton_G_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_SR_SelectMenuButton_GX_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_SR_SelectMenuButton_GY_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_SR_SelectMenuButton_GZ_CB( Widget wid, XtPointer client_data, XtPointer call_data );
void JC_SR_SelectMenuButton_GT_CB( Widget wid, XtPointer client_data, XtPointer call_data );

static void JC_SR_TransMenu_G_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_SR_TransMenu_GX_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_SR_TransMenu_GY_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_SR_TransMenu_GZ_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_SR_TransMenu_GT_CB( Widget wid, XtPointer client_data, XtPointer call_data );

static void JC_SR_MoreButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );
static void JC_SR_DismissButton_CB( Widget wid, XtPointer client_data, XtPointer call_data );

