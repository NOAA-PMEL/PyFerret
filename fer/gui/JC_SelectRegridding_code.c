/*
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
*/



/* 
 * JC_SelectRegridding_code.c
 *
 * Jonathan Callahan
 * Feb 6th 1996
 *
 * This file contains the auxiliary functions which are included by
 * JC_Regridding.c.
 *
 */

static JC_MenuItem items[MAX_MENU_ITEMS]={ NULL, };

/* .................... Function Definitions .................... */

static void JC_SR_TransMenu_G_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     strcpy(RG_ptr->rg_transform[ALL_AXES], (char *)client_data);
}


static void JC_SR_TransMenu_GX_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     strcpy(RG_ptr->rg_transform[X_AXIS], (char *)client_data);
}


static void JC_SR_TransMenu_GY_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     strcpy(RG_ptr->rg_transform[Y_AXIS], (char *)client_data);
}


static void JC_SR_TransMenu_GZ_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     strcpy(RG_ptr->rg_transform[Z_AXIS], (char *)client_data);
}


static void JC_SR_TransMenu_GT_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     strcpy(RG_ptr->rg_transform[T_AXIS], (char *)client_data);
}


void JC_SR_Initialize( void )
{

     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     char tempText[MAX_NAME_LENGTH]="";
     int small=160, large=250;
     int xyzt=0;

     *RG_ptr = GLOBAL_Regridding;

     JC_RegriddingWidgets_Init();
     JC_SelectRegridding_is_displayed = TRUE;

     sprintf(tempText, "Regrid %s of %s to", V_ptr->name, V_ptr->dset);
     XtVaSetValues(UxGetWidget(label1),
		   RES_CONVERT(XmNlabelString,  tempText),
		   NULL);
     for ( xyzt=0; xyzt<5; xyzt++ ) {
	  SelectMenu_widget[xyzt] = XmCreateMenuBar(rowColumn_widget[xyzt], "menubar", NULL, 0);
	  if ( xyzt == X_AXIS )
	       JC_SR_SelectMenu_Build(&(SelectMenu_widget[xyzt]), JC_SR_SelectMenuButton_GX_CB, JC_SR_SelectMenuButton_GX_CB, JC_SR_SelectMenuButton_GX_CB);
	  else if ( xyzt == Y_AXIS )
	       JC_SR_SelectMenu_Build(&(SelectMenu_widget[xyzt]), JC_SR_SelectMenuButton_GY_CB, JC_SR_SelectMenuButton_GY_CB, JC_SR_SelectMenuButton_GY_CB);
	  else if ( xyzt == Z_AXIS )
	       JC_SR_SelectMenu_Build(&(SelectMenu_widget[xyzt]), JC_SR_SelectMenuButton_GZ_CB, JC_SR_SelectMenuButton_GZ_CB, JC_SR_SelectMenuButton_GZ_CB);
	  else if ( xyzt == T_AXIS )
	       JC_SR_SelectMenu_Build(&(SelectMenu_widget[xyzt]), JC_SR_SelectMenuButton_GT_CB, JC_SR_SelectMenuButton_GT_CB, JC_SR_SelectMenuButton_GT_CB);
	  else if ( xyzt == ALL_AXES )
	       JC_SR_SelectMenu_Build(&(SelectMenu_widget[xyzt]), JC_SR_SelectMenuButton_G_CB, JC_SR_SelectMenuButton_G_CB, JC_SR_SelectMenuButton_G_CB);
	  XtManageChild(SelectMenu_widget[xyzt]);
	  XtManageChild(rowColumn_widget[xyzt]);

	  if ( !strcmp(RG_ptr->var[xyzt], "") )
	       strcpy(RG_ptr->var[xyzt], V_ptr->name);
	  if ( !strcmp(RG_ptr->dset[xyzt], "") )
	       strcpy(RG_ptr->dset[xyzt], V_ptr->dset);
	  sprintf(tempText, RG_ptr->var[xyzt]);
	  strcat(tempText, " from ");
	  strcat(tempText, RG_ptr->dset[xyzt]);
	  XmTextSetString(textField_widget[xyzt], tempText);
	  JC_SR_TransMenu_Initialize(RG_ptr, xyzt);
     }


     if ( RG_ptr->type == UNIFORM ) {
	  XtVaSetValues(UxGetWidget(form1),
			XmNheight, small,
			NULL);
	  XtUnmanageChild(UxGetWidget(form_non_uniform));
	  JC_NonUniform_Setup(R_ptr, RG_ptr);
	  XtManageChild(UxGetWidget(form_uniform));
	  XtVaSetValues(UxGetWidget(pushButton_More),
			RES_CONVERT(XmNlabelString,  "More..."),
			NULL);
     } else if ( RG_ptr->type == NON_UNIFORM ) {
	  XtVaSetValues(UxGetWidget(form1),
			XmNheight, large,
			NULL);
	  XtUnmanageChild(UxGetWidget(form_uniform));
	  JC_NonUniform_Setup(R_ptr, RG_ptr);
	  XtManageChild(UxGetWidget(form_non_uniform));
	  XtVaSetValues(UxGetWidget(pushButton_More),
			RES_CONVERT(XmNlabelString,  "Less..."),
			NULL);
     }
}


/*
 * This routine is for the first SelectMenu_Build.
 * All others use SelectMenu_Recreate in JC_InterInterface.c.
 *
 * The reason is that the Recreate version destroys and recreates the rowColumn container widget.
 */
static void JC_SR_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)()  )
{
     JC_Menu_AddDsetVars(items, var_fn_ptr, dvar_fn_ptr, cvar_fn_ptr);
     JC_Menu_Build(*menubar, XmMENU_PULLDOWN, "Select", NULL, FALSE, items);

}


static void JC_SR_TransMenu_Initialize( JC_Regridding *RG_ptr, int xyzt )
{
     if ( !strcmp(RG_ptr->rg_transform[xyzt], "") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_LIN],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "LIN") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_LIN],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "AVE") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_AVE],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "ASN") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_ASN],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "VAR") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_VAR],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "NGD") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_NGD],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "SUM") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_SUM],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "MIN") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_MIN],
			NULL);
     } else if ( !strcmp(RG_ptr->rg_transform[xyzt], "MAX") ) {
	  XtVaSetValues(TransMenu_widget[xyzt],
			XmNmenuHistory, TransMenuButton_widget[xyzt][RG_MAX],
			NULL);
     } else
	  fprintf(stderr, "ERROR in JC_SelectRegridding_code: JC_SR_TransMenu_Initialize: transform \"%s\" unknown.\n",
RG_ptr->rg_transform[xyzt]);

}


static void JC_SR_MoreButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Region *R_ptr=&GLOBAL_Region;
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     int small=160, large=250;

     if ( RG_ptr->type == UNIFORM ) {
	  XtVaSetValues(UxGetWidget(form1),
			XmNheight, large,
			NULL);
	  RG_ptr->type = NON_UNIFORM;
	  XtUnmanageChild(UxGetWidget(form_uniform));
	  XtManageChild(UxGetWidget(form_non_uniform));
	  XtVaSetValues(UxGetWidget(pushButton_More),
			RES_CONVERT(XmNlabelString,  "Less..."),
			NULL);
	  JC_SelectRegridding_is_uniform = FALSE;
     } else if ( RG_ptr->type == NON_UNIFORM ) {
	  XtVaSetValues(UxGetWidget(form1),
			XmNheight, small,
			NULL);
	  RG_ptr->type = UNIFORM;
	  XtUnmanageChild(UxGetWidget(form_non_uniform));
	  XtManageChild(UxGetWidget(form_uniform));
	  XtVaSetValues(UxGetWidget(pushButton_More),
			RES_CONVERT(XmNlabelString,  "More..."),
			NULL);
	  JC_SelectRegridding_is_uniform = TRUE;
     }

     JC_II_ChangeRegriddingLabel(JC_SelectRegridding);

}


void JC_SR_OKButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_II_FixRegridding(JC_SelectRegridding);
     XtPopdown(UxGetWidget(JC_SelectRegridding));
     JC_SelectRegridding_is_displayed = FALSE;
}


static void JC_SR_DismissButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     XtPopdown(UxGetWidget(JC_SelectRegridding));
     JC_SelectRegridding_is_displayed = FALSE;
}


static void JC_NonUniform_Setup( JC_Region *R_ptr, JC_Regridding *RG_ptr )
{
     int xyzt;

     for ( xyzt=0; xyzt<4; xyzt++ ) {
	  if ( R_ptr->span[xyzt].ss[LO] == IRRELEVANT_AXIS ) {
	       XtUnmanageChild(rowColumn_widget[xyzt]);
	       XtUnmanageChild(textField_widget[xyzt]);
	       XtUnmanageChild(TransMenu_widget[xyzt]);
	  } else {
	       XtManageChild(textField_widget[xyzt]);
	       XtManageChild(TransMenu_widget[xyzt]);
	  }
     }
}


void JC_SR_SelectMenuButton_G_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     char *tempText;
     char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]="", string[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;

     strcpy(dset_name, (char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(var_name, tempText);

     strcpy(RG_ptr->var[ALL_AXES], var_name);
     strcpy(RG_ptr->dset[ALL_AXES], dset_name);
     sprintf(string, "%s from %s", var_name, dset_name);
     XmTextSetString(textField_widget[ALL_AXES], string);

     XmStringFree(buttonLabel);
     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


void JC_SR_SelectMenuButton_GX_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     char *tempText;
     char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]="", string[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;

     strcpy(dset_name, (char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(var_name, tempText);

     strcpy(RG_ptr->var[X_AXIS], var_name);
     strcpy(RG_ptr->dset[X_AXIS], dset_name);
     sprintf(string, "%s from %s", var_name, dset_name);
     XmTextSetString(textField_widget[X_AXIS], string);

     XmStringFree(buttonLabel);
     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


void JC_SR_SelectMenuButton_GY_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     char *tempText;
     char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]="", string[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;

     strcpy(dset_name, (char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(var_name, tempText);

     strcpy(RG_ptr->var[Y_AXIS], var_name);
     strcpy(RG_ptr->dset[Y_AXIS], dset_name);
     sprintf(string, "%s from %s", var_name, dset_name);
     XmTextSetString(textField_widget[Y_AXIS], string);

     XmStringFree(buttonLabel);
     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


void JC_SR_SelectMenuButton_GZ_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     char *tempText;
     char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]="", string[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;

     strcpy(dset_name, (char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(var_name, tempText);

     strcpy(RG_ptr->var[Z_AXIS], var_name);
     strcpy(RG_ptr->dset[Z_AXIS], dset_name);
     sprintf(string, "%s from %s", var_name, dset_name);
     XmTextSetString(textField_widget[Z_AXIS], string);

     XmStringFree(buttonLabel);
     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


void JC_SR_SelectMenuButton_GT_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_Regridding *RG_ptr=&GLOBAL_Regridding;
     char *tempText;
     char dset_name[MAX_NAME_LENGTH]="", var_name[MAX_NAME_LENGTH]="", string[MAX_NAME_LENGTH]=""; 
     XmString buttonLabel;

     strcpy(dset_name, (char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &buttonLabel,
		   NULL);
     XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);
     strcpy(var_name, tempText);

     strcpy(RG_ptr->var[T_AXIS], var_name);
     strcpy(RG_ptr->dset[T_AXIS], dset_name);
     sprintf(string, "%s from %s", var_name, dset_name);
     XmTextSetString(textField_widget[T_AXIS], string);

     XmStringFree(buttonLabel);
     XtFree(tempText); /* allocated with XmStringGetLtoR() */
}


static void JC_RegriddingWidgets_Init( void )
{
     rowColumn_widget[X_AXIS] = UxGetWidget(rowColumn_Select_GX);
     textField_widget[X_AXIS] = UxGetWidget(textField_GX);
     TransMenu_widget[X_AXIS] = UxGetWidget(optionMenu_TransformGX);
     rowColumn_widget[Y_AXIS] = UxGetWidget(rowColumn_Select_GY);
     textField_widget[Y_AXIS] = UxGetWidget(textField_GY);
     TransMenu_widget[Y_AXIS] = UxGetWidget(optionMenu_TransformGY);
     rowColumn_widget[Z_AXIS] = UxGetWidget(rowColumn_Select_GZ);
     textField_widget[Z_AXIS] = UxGetWidget(textField_GZ);
     TransMenu_widget[Z_AXIS] = UxGetWidget(optionMenu_TransformGZ);
     rowColumn_widget[T_AXIS] = UxGetWidget(rowColumn_Select_GT);
     textField_widget[T_AXIS] = UxGetWidget(textField_GT);
     TransMenu_widget[T_AXIS] = UxGetWidget(optionMenu_TransformGT);

     rowColumn_widget[ALL_AXES] = UxGetWidget(rowColumn_Select_G);
     textField_widget[ALL_AXES] = UxGetWidget(textField_G);
     TransMenu_widget[ALL_AXES] = UxGetWidget(optionMenu_Transform_G);

     TransMenuButton_widget[X_AXIS][RG_LIN] = UxGetWidget(optionMenu_GX_LIN);
     TransMenuButton_widget[X_AXIS][RG_AVE] = UxGetWidget(optionMenu_GX_AVE);
     TransMenuButton_widget[X_AXIS][RG_ASN] = UxGetWidget(optionMenu_GX_ASN);
     TransMenuButton_widget[X_AXIS][RG_VAR] = UxGetWidget(optionMenu_GX_VAR);
     TransMenuButton_widget[X_AXIS][RG_NGD] = UxGetWidget(optionMenu_GX_NGD);
     TransMenuButton_widget[X_AXIS][RG_SUM] = UxGetWidget(optionMenu_GX_SUM);
     TransMenuButton_widget[X_AXIS][RG_MIN] = UxGetWidget(optionMenu_GX_MIN);
     TransMenuButton_widget[X_AXIS][RG_MAX] = UxGetWidget(optionMenu_GX_MAX);

     TransMenuButton_widget[Y_AXIS][RG_LIN] = UxGetWidget(optionMenu_GY_LIN);
     TransMenuButton_widget[Y_AXIS][RG_AVE] = UxGetWidget(optionMenu_GY_AVE);
     TransMenuButton_widget[Y_AXIS][RG_ASN] = UxGetWidget(optionMenu_GY_ASN);
     TransMenuButton_widget[Y_AXIS][RG_VAR] = UxGetWidget(optionMenu_GY_VAR);
     TransMenuButton_widget[Y_AXIS][RG_NGD] = UxGetWidget(optionMenu_GY_NGD);
     TransMenuButton_widget[Y_AXIS][RG_SUM] = UxGetWidget(optionMenu_GY_SUM);
     TransMenuButton_widget[Y_AXIS][RG_MIN] = UxGetWidget(optionMenu_GY_MIN);
     TransMenuButton_widget[Y_AXIS][RG_MAX] = UxGetWidget(optionMenu_GY_MAX);

     TransMenuButton_widget[Y_AXIS][RG_LIN] = UxGetWidget(optionMenu_GZ_LIN);
     TransMenuButton_widget[Y_AXIS][RG_AVE] = UxGetWidget(optionMenu_GZ_AVE);
     TransMenuButton_widget[Y_AXIS][RG_ASN] = UxGetWidget(optionMenu_GZ_ASN);
     TransMenuButton_widget[Y_AXIS][RG_VAR] = UxGetWidget(optionMenu_GZ_VAR);
     TransMenuButton_widget[Y_AXIS][RG_NGD] = UxGetWidget(optionMenu_GZ_NGD);
     TransMenuButton_widget[Y_AXIS][RG_SUM] = UxGetWidget(optionMenu_GZ_SUM);
     TransMenuButton_widget[Y_AXIS][RG_MIN] = UxGetWidget(optionMenu_GZ_MIN);
     TransMenuButton_widget[Y_AXIS][RG_MAX] = UxGetWidget(optionMenu_GZ_MAX);

     TransMenuButton_widget[Z_AXIS][RG_LIN] = UxGetWidget(optionMenu_GT_LIN);
     TransMenuButton_widget[Z_AXIS][RG_AVE] = UxGetWidget(optionMenu_GT_AVE);
     TransMenuButton_widget[Z_AXIS][RG_ASN] = UxGetWidget(optionMenu_GT_ASN);
     TransMenuButton_widget[Z_AXIS][RG_VAR] = UxGetWidget(optionMenu_GT_VAR);
     TransMenuButton_widget[Z_AXIS][RG_NGD] = UxGetWidget(optionMenu_GT_NGD);
     TransMenuButton_widget[Z_AXIS][RG_SUM] = UxGetWidget(optionMenu_GT_SUM);
     TransMenuButton_widget[Z_AXIS][RG_MIN] = UxGetWidget(optionMenu_GT_MIN);
     TransMenuButton_widget[Z_AXIS][RG_MAX] = UxGetWidget(optionMenu_GT_MAX);

     TransMenuButton_widget[ALL_AXES][RG_LIN] = UxGetWidget(optionMenu_p1_LIN);
     TransMenuButton_widget[ALL_AXES][RG_AVE] = UxGetWidget(optionMenu_p1_AVE);
     TransMenuButton_widget[ALL_AXES][RG_ASN] = UxGetWidget(optionMenu_p1_ASN);
     TransMenuButton_widget[ALL_AXES][RG_VAR] = UxGetWidget(optionMenu_p1_VAR);
     TransMenuButton_widget[ALL_AXES][RG_NGD] = UxGetWidget(optionMenu_p1_NGD);
     TransMenuButton_widget[ALL_AXES][RG_SUM] = UxGetWidget(optionMenu_p1_SUM);
     TransMenuButton_widget[ALL_AXES][RG_MIN] = UxGetWidget(optionMenu_p1_MIN);
     TransMenuButton_widget[ALL_AXES][RG_MAX] = UxGetWidget(optionMenu_p1_MAX);
}



