/* 
 * JC_DefineVariable_code.c
 *
 * Jonathan Callahan
 * Feb 13th 1996
 *
 * This file contains the auxiliary functions which are included by
 * JC_DefineVariable.c.
 *
 */

/* .................... Function Definitions .................... */


static void JC_DV_Initialize( int function_type )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     JC_Variable *V_ptr=&GLOBAL_Variable;
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     JC_Object *O_ptr=NULL;
     char tempText[MAX_NAME_LENGTH]="";
     JC_DatasetElement *DE_ptr=NULL;
     int i=1;

     GLOBAL_func_type = function_type;

     JC_DefineVariable_is_displayed = TRUE;
     JC_II_MainMenu_Maintain( SF_ptr );
     
     list_traverse(GLOBAL_DatasetList, V_ptr->dset, JC_ListTraverse_FoundDsetMatch, (LIST_FRNT | LIST_FORW | LIST_ALTR));
     DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);

     sprintf(tempText, "DVAR_%d", i++);
     while ( JC_DatasetElement_VarnameExists(DE_ptr, tempText) ) {
	  sprintf(tempText, "DVAR_%d", i++);
     }
     
/*
 * For cloned variables we need to get the actual 'variable' name
 */
     if ( SF_ptr->a_clone_is_selected ) {
	  O_ptr = SF_ptr->current_clone_ptr;
	  strcpy(DV_ptr->assigned_dset, O_ptr->variable.dset);
	  strcpy(DV_ptr->var[0], O_ptr->variable.name);
	  strcpy(DV_ptr->dset[0], O_ptr->variable.dset);
	  DV_ptr->clone_ptr[0] = O_ptr;
	  strcpy(DV_ptr->var[1], O_ptr->variable.name);
	  strcpy(DV_ptr->dset[1], O_ptr->variable.dset);
	  DV_ptr->clone_ptr[1] = O_ptr;
     } else {
	  strcpy(DV_ptr->assigned_dset, V_ptr->dset);
	  strcpy(DV_ptr->var[0], V_ptr->name);
	  strcpy(DV_ptr->dset[0], V_ptr->dset);
	  DV_ptr->clone_ptr[0] = NULL;
	  strcpy(DV_ptr->var[1], V_ptr->name);
	  strcpy(DV_ptr->dset[1], V_ptr->dset);
	  DV_ptr->clone_ptr[0] = NULL;
     }
     
     strcpy(DV_ptr->name, tempText);
     strcpy(DV_ptr->multiplier[0], "1.0");
     strcpy(DV_ptr->multiplier[1], "1.0");
     strcpy(DV_ptr->operator, "+");	  
     strcpy(DV_ptr->function, "INT");	  
     DV_ptr->type = function_type;
     DV_ptr->number_of_vars = 1;
   
     switch (function_type) {

       /*
	* Note that DV_ptr->number_of_vars is not set here.  It is set when the function is selected.
	*/
	case FUNC_FUNCTION1:
	case FUNC_FUNCTION2:
	case FUNC_FUNCTION3:
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];
	  JC_DV_menu_datasets[1].label = NULL;
	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(rowColumn1));
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
	  XtManageChild(UxGetWidget(label_Title));
	  XtManageChild(UxGetWidget(textField_Title));
	  XtManageChild(UxGetWidget(label_Units));
	  XtManageChild(UxGetWidget(textField_Units));
     
	  JC_DV_FunctionMenu_F = JC_Menu_Build(rowColumn_F, XmMENU_OPTION, NULL, NULL, FALSE, JC_DV_menu_functions);
	  XtManageChild(UxGetWidget(rowColumn_F));
/*
	  XtVaSetValues(UxGetWidget(JC_DV_FunctionMenu_F),
			XmNmenuHistory, ???,
			NULL);
*/
	  XtVaSetValues(rowColumn1_F,
                        XmNpositionIndex, 0,
                        NULL );
	  XtVaSetValues(JC_DV_FunctionMenu_F,
                        XmNpositionIndex, 1,
                        NULL );
	  XtVaSetValues(rowColumn_var1_F,
                        XmNpositionIndex, 2,
                        NULL );
  	  XtManageChild(UxGetWidget(rowColumn1_F));
	  XtManageChild(UxGetWidget(JC_DV_FunctionMenu_F));
	  XtManageChild(UxGetWidget(rowColumn_var1_F));
	  XtManageChild(UxGetWidget(rowColumn_end_F));

	  JC_DV_SelectMenu1_F = XmCreateMenuBar(rowColumn_Select1_F, "menubar1F", NULL, 0);
	  JC_DV_SelectMenu_Build(&JC_DV_SelectMenu1_F, JC_DV_SelectMenuButton_F1_CB, JC_DV_SelectMenuButton_F1_CB, JC_DV_SelectMenuF1_CloneButton_CB);
	  XtManageChild(JC_DV_SelectMenu1_F);
	  XtManageChild(rowColumn_Select1_F);

	  JC_DV_SelectMenu2_F = XmCreateMenuBar(rowColumn_Select_var2, "menubar2F", NULL, 0);
	  JC_DV_SelectMenu_Build(&JC_DV_SelectMenu2_F, JC_DV_SelectMenuButton_F2_CB, JC_DV_SelectMenuButton_F2_CB, JC_DV_SelectMenuF2_CloneButton_CB);
/*
	  XtManageChild(JC_DV_SelectMenu2_F);
	  XtManageChild(rowColumn_Select_var2);
*/
	  XtManageChild(UxGetWidget(label_dset_var1));

	  XmTextSetString(textField_F_VarName, DV_ptr->name);
	  XmTextSetString(textField1_1var, DV_ptr->multiplier[0]);
	  XmTextSetString(textField2_1var2, DV_ptr->var[0]);
          XtVaSetValues(UxGetWidget(label_dset_var1),
                        RES_CONVERT(XmNlabelString,  DV_ptr->dset[0]),
                        NULL);
	  XmTextSetString(textField1_var2, DV_ptr->multiplier[1]);
	  XmTextSetString(textField2_var2, DV_ptr->var[1]);
          XtVaSetValues(UxGetWidget(label_dset_var2),
                        RES_CONVERT(XmNlabelString,  DV_ptr->dset[1]),
                        NULL);
	  break;

	case FUNC_LINEAR_COMBINATION:
	  DV_ptr->number_of_vars = 2;

	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];
	  JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	  JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  JC_DV_menu_datasets[2].label = NULL;
	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(rowColumn1));
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
	  XtManageChild(UxGetWidget(label_Title));
	  XtManageChild(UxGetWidget(textField_Title));
	  XtManageChild(UxGetWidget(label_Units));
	  XtManageChild(UxGetWidget(textField_Units));
	  
	  JC_DV_OperatorMenu_LC = JC_Menu_Build(rowColumn_LC, XmMENU_OPTION, NULL, NULL, FALSE, JC_DV_menu_operators_LC);
	  XtManageChild(UxGetWidget(rowColumn_LC));
	  XtVaSetValues(rowColumn1_LC,
                        XmNpositionIndex, 0,
                        NULL );
	  XtVaSetValues(JC_DV_OperatorMenu_LC,
                        XmNpositionIndex, 1,
                        NULL );
	  XtVaSetValues(rowColumn2_LC,
                        XmNpositionIndex, 2,
                        NULL );
	  XtManageChild(UxGetWidget(rowColumn1_LC));
	  XtManageChild(UxGetWidget(JC_DV_OperatorMenu_LC));
	  XtManageChild(UxGetWidget(rowColumn2_LC));

	  JC_DV_SelectMenu1_LC = XmCreateMenuBar(rowColumn_Select1_LC, "menubar1LC", NULL, 0);
	  JC_DV_SelectMenu_Build(&JC_DV_SelectMenu1_LC, JC_DV_SelectMenuButton_LC1_CB, JC_DV_SelectMenuButton_LC1_CB, JC_DV_SelectMenuLC1_CloneButton_CB);
	  XtManageChild(JC_DV_SelectMenu1_LC);
	  XtManageChild(rowColumn_Select1_LC);
	  XtManageChild(UxGetWidget(label_dset1_LC));

	  JC_DV_SelectMenu2_LC = XmCreateMenuBar(rowColumn_Select2_LC, "menubar2LC", NULL, 0);
	  JC_DV_SelectMenu_Build(&JC_DV_SelectMenu2_LC, JC_DV_SelectMenuButton_LC2_CB, JC_DV_SelectMenuButton_LC2_CB, JC_DV_SelectMenuLC2_CloneButton_CB);
	  XtManageChild(JC_DV_SelectMenu2_LC);
	  XtManageChild(rowColumn_Select2_LC);
	  XtManageChild(UxGetWidget(label_dset2_LC));

	  XmTextSetString(textField_LC_VarName, DV_ptr->name);
	  XmTextSetString(textField_LC_1, DV_ptr->multiplier[0]);
	  XmTextSetString(textField_LC_3, DV_ptr->multiplier[1]);
	  XmTextSetString(textField_LC_2, DV_ptr->var[0]);
	  XmTextSetString(textField_LC_4, DV_ptr->var[1]);
          XtVaSetValues(UxGetWidget(label_dset1_LC),
                        RES_CONVERT(XmNlabelString,  DV_ptr->dset[0]),
                        NULL);
          XtVaSetValues(UxGetWidget(label_dset2_LC),
                        RES_CONVERT(XmNlabelString,  DV_ptr->dset[1]),
                        NULL);
	  break;

	case FUNC_PLUS_CONSTANT:
	  DV_ptr->number_of_vars = 1;
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];
	  JC_DV_menu_datasets[1].label = NULL;
	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(rowColumn1));
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
	  XtManageChild(UxGetWidget(label_Title));
	  XtManageChild(UxGetWidget(textField_Title));
	  XtManageChild(UxGetWidget(label_Units));
	  XtManageChild(UxGetWidget(textField_Units));
	  
	  JC_DV_OperatorMenu_EXP = JC_Menu_Build(rowColumn_EXP, XmMENU_OPTION, NULL, NULL, FALSE, JC_DV_menu_operators_EXP);
	  XtManageChild(UxGetWidget(rowColumn_EXP));
	  XtVaSetValues(rowColumn1_EXP,
                        XmNpositionIndex, 0,
                        NULL );
	  XtVaSetValues(JC_DV_OperatorMenu_EXP,
                        XmNpositionIndex, 1,
                        NULL );
	  XtVaSetValues(rowColumn2_EXP,
                        XmNpositionIndex, 2,
                        NULL );

	  JC_DV_SelectMenu1_EXP = XmCreateMenuBar(rowColumn_Select1_EXP, "menubar1EXP", NULL, 0);
	  JC_DV_SelectMenu_Build(&JC_DV_SelectMenu1_EXP, JC_DV_SelectMenuButton_EXP1_CB, JC_DV_SelectMenuButton_EXP1_CB, JC_DV_SelectMenuEXP1_CloneButton_CB);
	  XtManageChild(JC_DV_SelectMenu1_EXP);
	  XtManageChild(rowColumn_Select1_EXP);

	  XtManageChild(UxGetWidget(label_dset1_EXP));
	  XtManageChild(UxGetWidget(rowColumn1_EXP));
	  XtManageChild(UxGetWidget(JC_DV_OperatorMenu_EXP));
	  XtManageChild(UxGetWidget(rowColumn2_EXP));
	  XmTextSetString(textField1_EXP, DV_ptr->name);
	  XmTextSetString(textField2_EXP, DV_ptr->multiplier[0]);
	  XmTextSetString(textField3_EXP, DV_ptr->var[0]);
	  XmTextSetString(textField4_EXP, DV_ptr->multiplier[1]);
          XtVaSetValues(UxGetWidget(label_dset1_EXP),
                        RES_CONVERT(XmNlabelString,  DV_ptr->dset[0]),
                        NULL);
	  break;

	case FUNC_EXPONENT:
	  DV_ptr->number_of_vars = 1;
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];
	  JC_DV_menu_datasets[1].label = NULL;
	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(rowColumn1));
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
	  XtManageChild(UxGetWidget(label_Title));
	  XtManageChild(UxGetWidget(textField_Title));
	  XtManageChild(UxGetWidget(label_Units));
	  XtManageChild(UxGetWidget(textField_Units));
	  
	  JC_DV_OperatorMenu_EXP = JC_Menu_Build(rowColumn_EXP, XmMENU_OPTION, NULL, NULL, FALSE, JC_DV_menu_operators_EXP);
	  XtManageChild(UxGetWidget(rowColumn_EXP));
	  XtVaSetValues(rowColumn1_EXP,
                        XmNpositionIndex, 0,
                        NULL );
	  XtVaSetValues(JC_DV_OperatorMenu_EXP,
                        XmNpositionIndex, 1,
                        NULL );
	  XtVaSetValues(rowColumn2_EXP,
                        XmNpositionIndex, 2,
                        NULL );

	  JC_DV_SelectMenu1_EXP = XmCreateMenuBar(rowColumn_Select1_EXP, "menubar1EXP", NULL, 0);
	  JC_DV_SelectMenu_Build(&JC_DV_SelectMenu1_EXP, JC_DV_SelectMenuButton_EXP1_CB, JC_DV_SelectMenuButton_EXP1_CB, JC_DV_SelectMenuEXP1_CloneButton_CB);
	  XtManageChild(JC_DV_SelectMenu1_EXP);
	  XtManageChild(rowColumn_Select1_EXP);

	  XtManageChild(UxGetWidget(label_dset1_EXP));
	  XtManageChild(UxGetWidget(rowColumn1_EXP));
	  XtManageChild(UxGetWidget(JC_DV_OperatorMenu_EXP));
	  XtManageChild(UxGetWidget(rowColumn2_EXP));
	  XmTextSetString(textField1_EXP, DV_ptr->name);
	  XmTextSetString(textField2_EXP, DV_ptr->multiplier[0]);
	  XmTextSetString(textField3_EXP, DV_ptr->var[0]);
	  XmTextSetString(textField4_EXP, DV_ptr->multiplier[1]);
/*
	  XtVaSetValues(UxGetWidget(optionMenu_Operator1_EXP),
			XmNmenuHistory, optionMenu_EXPOp_p1_Exponentiate1,
			NULL);
*/	  
          XtVaSetValues(UxGetWidget(label_dset1_EXP),
                        RES_CONVERT(XmNlabelString,  DV_ptr->dset[0]),
                        NULL);
	  break;
	  
	default:
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_Initialize: function_type = %d\n", function_type);
	  
     }

}


static void JC_DV_DsetMenu_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;

     strcpy(DV_ptr->assigned_dset, (char *)client_data);
}


static void JC_DV_FunctionMenu_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     int j=0;

     strcpy(DV_ptr->function, (char *)client_data);
     
/*
 * - Functions of a single variable:
 */

     if ( !strcmp(DV_ptr->function, "INT")
	 || !strcmp(DV_ptr->function, "ABS")
	 || !strcmp(DV_ptr->function, "EXP")
	 || !strcmp(DV_ptr->function, "LN")
	 || !strcmp(DV_ptr->function, "LOG")
	 || !strcmp(DV_ptr->function, "SIN")
	 || !strcmp(DV_ptr->function, "COS")
	 || !strcmp(DV_ptr->function, "TAN")
	 || !strcmp(DV_ptr->function, "ASIN")
	 || !strcmp(DV_ptr->function, "ACOS")
	 || !strcmp(DV_ptr->function, "ATAN")
	 || !strcmp(DV_ptr->function, "IGNORE0")
	 || !strcmp(DV_ptr->function, "RANDU")
	 || !strcmp(DV_ptr->function, "RANDN")
	 ) {
	  
	  if ( GLOBAL_func_type == FUNC_FUNCTION1 )
	       return;
	  
	  else if ( GLOBAL_func_type == FUNC_FUNCTION2 ) {
	       XtUnmanageChild(UxGetWidget(rowColumn_end_F));
	       XtUnmanageChild(UxGetWidget(rowColumn_var2_F));
	       XtUnmanageChild(JC_DV_SelectMenu2_F);
	       XtUnmanageChild(rowColumn_Select_var2);
	       XtUnmanageChild(UxGetWidget(label_dset_var2));
	       XtVaSetValues(rowColumn_end_F,
			     XmNpositionIndex, 3,
			     NULL);
	       XtManageChild(UxGetWidget(rowColumn_end_F));
	       GLOBAL_func_type = FUNC_FUNCTION1;
	  }
	  
	  DV_ptr->type = GLOBAL_func_type;
	  DV_ptr->number_of_vars = 1;
     }
     
/*
 * - Functions of two variables:
 */

     if ( !strcmp(DV_ptr->function, "MAX")
	 || !strcmp(DV_ptr->function, "MIN")
	 || !strcmp(DV_ptr->function, "ATAN2")
	 || !strcmp(DV_ptr->function, "MOD")
	 || !strcmp(DV_ptr->function, "MISSING")
	 ) {
	  
	  if ( GLOBAL_func_type == FUNC_FUNCTION2 )
	       return;
	  
	  else if ( GLOBAL_func_type == FUNC_FUNCTION1 ) {
	       XtUnmanageChild(UxGetWidget(rowColumn_end_F));
	       XtVaSetValues(rowColumn_var2_F,
			     XmNpositionIndex, 3,
			     NULL );
	       XtManageChild(UxGetWidget(rowColumn_var2_F));
	       XtManageChild(UxGetWidget(rowColumn_end_F));
	       XtManageChild(JC_DV_SelectMenu2_F);
	       XtManageChild(rowColumn_Select_var2);
	       XtManageChild(UxGetWidget(label_dset_var2));
	       GLOBAL_func_type = FUNC_FUNCTION2;
	  }
/*
	  else if ( GLOBAL_func_type == FUNC_FUNCTION3 ) {
	     ;
	  }
*/
	  DV_ptr->type = GLOBAL_func_type;
	  DV_ptr->number_of_vars = 2;
     }
     
/*
 * - Functions of three variables:
 */
/*
     if ( !strcmp(DV_ptr->function, "RHO_UN") ) {

	  if ( GLOBAL_func_type == FUNC_FUNCTION1 ) {
	       XtUnmanageChild(UxGetWidget(rowColumn_end_F));
	       XtVaSetValues(rowColumn_var2_F,
			     XmNpositionIndex, 3,
			     NULL );
	       XtManageChild(UxGetWidget(rowColumn_var2_F));
	       XtManageChild(UxGetWidget(rowColumn_end_F));
	  }

	  else if ( GLOBAL_func_type == FUNC_FUNCTION3 ) {
	       ;
	  }

	  DV_ptr->type = GLOBAL_func_type;
	  DV_ptr->number_of_vars = 3;
     }
*/

}


static void JC_DV_OperatorMenu_LC_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;

     strcpy(DV_ptr->operator, (char *) client_data);
}


static void JC_DV_OperatorMenu_EXP_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;

     strcpy(DV_ptr->operator, (char *) client_data);
}


static void JC_DV_Button1_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char my_test_command[MAX_COMMAND_LENGTH]="";
     char *text;

     JC_DatasetElement *DE_ptr;
     int status = LIST_OK;

     if ( text=(char *)XmTextGetString(textField_Title) )
	  strcpy(DV_ptr->title, text);
     
     if ( text=(char *)XmTextGetString(textField_Units) )
	  strcpy(DV_ptr->units, text);
     
     switch (GLOBAL_func_type) {


	case FUNC_FUNCTION1:

	  if ( text=(char *)XmTextGetString(textField_F_VarName) ) {
	       strcpy(DV_ptr->name, text);
	   } else {
/* JC_TODO ... I should be generating an appropriate name here */
	       strcpy(DV_ptr->name, "UserVariable_1");
	  }

	  if ( text=(char *)XmTextGetString(textField1_1var) ) {
	       strcpy(DV_ptr->multiplier[0], text);
	  } else {
	       strcpy(DV_ptr->multiplier[0], "1.0");
	  }
	  break;


	case FUNC_FUNCTION2:

	  if ( text=(char *)XmTextGetString(textField_F_VarName) ) {
	       strcpy(DV_ptr->name, text);
	   } else {
/* JC_TODO ... I should be generating an appropriate name here */
	       strcpy(DV_ptr->name, "UserVariable_1");
	  }

	  if ( text=(char *)XmTextGetString(textField1_1var) ) {
	       strcpy(DV_ptr->multiplier[0], text);
	  } else {
	       strcpy(DV_ptr->multiplier[0], "1.0");
	  }

	  if ( text=(char *)XmTextGetString(textField1_var2) ) {
	       strcpy(DV_ptr->multiplier[1], text);
	  } else {
	       strcpy(DV_ptr->multiplier[1], "1.0");
	  }
	  break;


	case FUNC_LINEAR_COMBINATION:

	  if ( text=(char *)XmTextGetString(textField_LC_VarName) ) {
	       strcpy(DV_ptr->name, text);
	  } else {
/* JC_TODO ... I should be generating an appropriate name here */
	       strcpy(DV_ptr->name, "UserVariable_1");
	  }

	  if ( text=(char *)XmTextGetString(textField_LC_1) ) {
	       strcpy(DV_ptr->multiplier[0], text);
	  } else {
	       strcpy(DV_ptr->multiplier[0], "1.0");
	  }

	  if ( text=(char *)XmTextGetString(textField_LC_3) ) {
	       strcpy(DV_ptr->multiplier[1], text);
	  } else {
	       strcpy(DV_ptr->multiplier[1], "1.0");
	  }
	  break;

	case FUNC_PLUS_CONSTANT:
	case FUNC_EXPONENT:
	  if ( text=(char *)XmTextGetString(textField1_EXP) ) {
	       strcpy(DV_ptr->name, text);
	  } else {
/* JC_TODO ... I should be generating an appropriate name here */
	       strcpy(DV_ptr->name, "UserVariable_1");
	  }

	  if ( text=(char *)XmTextGetString(textField2_EXP) ) {
	       strcpy(DV_ptr->multiplier[0], text);
	  } else {
	       strcpy(DV_ptr->multiplier[0], "1.0");
	  }

	  if ( text=(char *)XmTextGetString(textField4_EXP) ) {
	       strcpy(DV_ptr->multiplier[1], text);
	  } else {
	       strcpy(DV_ptr->multiplier[1], "1.0");
	  }
	  break;

	default:
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_Button1_CB(): GLOBAL_func_type = %d\n", GLOBAL_func_type);

     }

/*
 * - Create the LET command.
 * - Issue the command to Ferret.
 * - Get the current DatasetElement.
 * - 
 * - Query Ferret about the this dataset so that this DatasetElement reflects the newly defined variable.
 * - Recreate all the Select menus.
 */

     JC_LetCommand_Create(my_test_command, DV_ptr);
     ferret_command(my_test_command, IGNORE_COMMAND_WIDGET);
     list_traverse(GLOBAL_DatasetList, DV_ptr->assigned_dset, JC_ListTraverse_FoundDsetMatch,
		   (LIST_FRNT | LIST_FORW | LIST_ALTR));
     DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);
     JC_DatasetElement_QueryFerret(DE_ptr, FALSE);
     JC_II_SelectMenus_Recreate(UxGetWidget(JC_DefineVariable));
     
}


static void JC_DV_Button3_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;

     XtPopdown(UxGetWidget(JC_DefineVariable));
     JC_DefineVariable_is_displayed = FALSE;
     JC_II_MainMenu_Maintain( SF_ptr );
}


/*
 * This routine is for the first SelectMenu_Build.
 * All others use SelectMenu_Recreate in JC_InterInterface.c.
 *
 * The reason is that the Recreate version destroys and recreates the rowColumn container widget.
 */
void JC_DV_SelectMenu_Build( Widget *menubar, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)()  )
{
     JC_MenuItem items[MAX_MENU_ITEMS]={ NULL, };

     JC_Menu_AddDsetVars(items, var_fn_ptr, dvar_fn_ptr, cvar_fn_ptr);
     JC_Menu_Build(*menubar, XmMENU_PULLDOWN, "Select", NULL, FALSE, items);

}


void JC_DV_SelectMenuButton_F1_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     String var, dset;
     XmString label;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[0]) ) {
	  strcpy(DV_ptr->dset[0], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[0], var);
     strcpy(DV_ptr->dset[0], dset);
     DV_ptr->clone_ptr[0] = NULL;
     XmTextSetString(textField2_1var2, var);
     XtVaSetValues(UxGetWidget(label_dset_var1),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuF1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char var_expression[MAX_NAME_LENGTH]="";
     String var, dset;
     XmString label;
     JC_Object *O_ptr=NULL;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuF1_CloneButton_CB: XmStringGetLtoR failed.\n");

/*
 * For cloned variables we need to look up the name in the cvarList and get the actual 'variable' name
 */
     O_ptr = JC_Clone_ReturnPointer(var, dset);
     strcpy(var, O_ptr->variable.name);

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[0]) ) {
	  strcpy(DV_ptr->dset[0], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[0], var);
     strcpy(DV_ptr->dset[0], dset);
     strcpy(DV_ptr->assigned_dset, O_ptr->variable.dset);
     DV_ptr->clone_ptr[0] = O_ptr;
     JC_LetCommand_CreateClonedVarExpression(var_expression, 0, DV_ptr);
     XmTextSetString(textField2_1var2, var_expression);
     XtVaSetValues(UxGetWidget(label_dset_var1),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuButton_F2_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     String var, dset;
     XmString label;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[1]) ) {
	  strcpy(DV_ptr->dset[1], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[1];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[1], var);
     strcpy(DV_ptr->dset[1], dset);
     DV_ptr->clone_ptr[1] = NULL;
     XmTextSetString(textField2_var2, var);
     XtVaSetValues(UxGetWidget(label_dset_var2),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuF2_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char var_expression[MAX_NAME_LENGTH]="";
     String var, dset;
     XmString label;
     JC_Object *O_ptr=NULL;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuF1_CloneButton_CB: XmStringGetLtoR failed.\n");

/*
 * For cloned variables we need to look up the name in the cvarList and get the actual 'variable' name
 */
     O_ptr = JC_Clone_ReturnPointer(var, dset);
     strcpy(var, O_ptr->variable.name);

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[1]) ) {
	  strcpy(DV_ptr->dset[1], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[1];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[1], var);
     strcpy(DV_ptr->dset[1], dset);
     strcpy(DV_ptr->assigned_dset, O_ptr->variable.dset);
     DV_ptr->clone_ptr[1] = O_ptr;
     JC_LetCommand_CreateClonedVarExpression(var_expression, 1, DV_ptr);
     XmTextSetString(textField2_var2, var_expression);
     XtVaSetValues(UxGetWidget(label_dset_var2),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuButton_F3_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     String var, dset;
     XmString label;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/* Set values */
     strcpy(DV_ptr->var[2], var);
     strcpy(DV_ptr->dset[2], dset);
     DV_ptr->clone_ptr[2] = NULL;
     XmTextSetString(textField2_var3, var);
     XtVaSetValues(UxGetWidget(label_dset_var3),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuF3_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char var_expression[MAX_NAME_LENGTH]="";
     String var, dset;
     XmString label;
     JC_Object *O_ptr=NULL;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/*
 * For cloned variables we need to look up the name in the cvarList and get the actual 'variable' name
 */
     O_ptr = JC_Clone_ReturnPointer(var, dset);
     strcpy(var, O_ptr->variable.name);

/* Set values */
     strcpy(DV_ptr->var[2], var);
     strcpy(DV_ptr->dset[2], dset);
     strcpy(DV_ptr->assigned_dset, O_ptr->variable.dset);
     DV_ptr->clone_ptr[2] = O_ptr;
     JC_LetCommand_CreateClonedVarExpression(var_expression, 2, DV_ptr);
     XmTextSetString(textField2_var3, var_expression);
     XtVaSetValues(UxGetWidget(label_dset_var3),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuButton_LC1_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     String var, dset;
     XmString label;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[0]) ) {
	  strcpy(DV_ptr->dset[0], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[1];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[0], var);
     strcpy(DV_ptr->dset[0], dset);
     DV_ptr->clone_ptr[0] = NULL;
     XmTextSetString(textField_LC_2, var);
     XtVaSetValues(UxGetWidget(label_dset1_LC),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuLC1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char var_expression[MAX_NAME_LENGTH]="";
     String var, dset;
     XmString label;
     JC_Object *O_ptr=NULL;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/*
 * For cloned variables we need to look up the name in the cvarList and get the actual 'variable' name
 */
     O_ptr = JC_Clone_ReturnPointer(var, dset);
     strcpy(var, O_ptr->variable.name);

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[0]) ) {
	  strcpy(DV_ptr->dset[0], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[1];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[0], var);
     strcpy(DV_ptr->dset[0], dset);
     strcpy(DV_ptr->assigned_dset, O_ptr->variable.dset);
     DV_ptr->clone_ptr[0] = O_ptr;
     JC_LetCommand_CreateClonedVarExpression(var_expression, 0, DV_ptr);
     XmTextSetString(textField_LC_2, var_expression);
     XtVaSetValues(UxGetWidget(label_dset1_LC),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuButton_LC2_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     String var, dset;
     XmString label;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[1]) ) {
	  strcpy(DV_ptr->dset[1], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[1];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[1], var);
     strcpy(DV_ptr->dset[1], dset);
     DV_ptr->clone_ptr[1] = NULL;
     XmTextSetString(textField_LC_4, var);
     XtVaSetValues(UxGetWidget(label_dset2_LC),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuLC2_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char var_expression[MAX_NAME_LENGTH]="";
     String var, dset;
     XmString label;
     JC_Object *O_ptr=NULL;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/*
 * For cloned variables we need to look up the name in the cvarList and get the actual 'variable' name
 */
     O_ptr = JC_Clone_ReturnPointer(var, dset);
     strcpy(var, O_ptr->variable.name);

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[1]) ) {
	  strcpy(DV_ptr->dset[1], dset);
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[1];

	  if ( strcmp(DV_ptr->dset[1], DV_ptr->dset[0]) ) {
	       JC_DV_menu_datasets[1].label = DV_ptr->dset[1];
	       JC_DV_menu_datasets[1].class = &xmPushButtonWidgetClass;
	       JC_DV_menu_datasets[1].callback = JC_DV_DsetMenu_CB;
	       JC_DV_menu_datasets[1].callback_data = (XtPointer) DV_ptr->dset[1];
	  } else
	       JC_DV_menu_datasets[1].label = NULL;

	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[1], var);
     strcpy(DV_ptr->dset[1], dset);
     strcpy(DV_ptr->assigned_dset, O_ptr->variable.dset);
     DV_ptr->clone_ptr[1] = O_ptr;
     JC_LetCommand_CreateClonedVarExpression(var_expression, 1, DV_ptr);
     XmTextSetString(textField_LC_4, var_expression);
     XtVaSetValues(UxGetWidget(label_dset2_LC),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuButton_EXP1_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char var_expression[MAX_NAME_LENGTH]="";
     String var, dset;
     XmString label;
     JC_Object *O_ptr=NULL;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[0]) ) {
	  strcpy(DV_ptr->dset[0], dset);
	  XtUnmanageChild(UxGetWidget(JC_DV_DsetMenu));
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];
	  JC_DV_menu_datasets[1].label = NULL;
	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[0], var);
     strcpy(DV_ptr->dset[0], dset);
     DV_ptr->clone_ptr[0] = NULL;
     XmTextSetString(textField3_EXP, var);
     XtVaSetValues(UxGetWidget(label_dset1_EXP),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


void JC_DV_SelectMenuEXP1_CloneButton_CB( Widget wid, XtPointer client_data, XtPointer call_data )
{
     JC_DefinedVariable *DV_ptr=&my_test_DV;
     char var_expression[MAX_NAME_LENGTH]="";
     String var, dset;
     XmString label;
	JC_Object *O_ptr=NULL;

/* Get values */
     dset = XtNewString((char *)client_data);
     XtVaGetValues(wid,
		   XmNlabelString, &label,
		   NULL);
     if ( !XmStringGetLtoR(label, XmSTRING_DEFAULT_CHARSET, &var) )
	  fprintf(stderr, "ERROR in JC_DefineVariable_code.c: JC_DV_SelectMenuButton_F1_CB: XmStringGetLtoR failed.\n");

/*
 * For cloned variables we need to look up the name in the cvarList and get the actual 'variable' name
 */
     O_ptr = JC_Clone_ReturnPointer(var, dset);
     strcpy(var, O_ptr->variable.name);

/* Recreate the DsetMenu if necessary */
     if ( strcmp(dset, DV_ptr->dset[0]) ) {
	  strcpy(DV_ptr->dset[0], dset);
	  XtUnmanageChild(UxGetWidget(JC_DV_DsetMenu));
	  XtDestroyWidget(UxGetWidget(JC_DV_DsetMenu));
	  JC_DV_menu_datasets[0].label = DV_ptr->dset[0];
	  JC_DV_menu_datasets[0].class = &xmPushButtonWidgetClass;
	  JC_DV_menu_datasets[0].callback = JC_DV_DsetMenu_CB;
	  JC_DV_menu_datasets[0].callback_data = (XtPointer) DV_ptr->dset[0];
	  JC_DV_menu_datasets[1].label = NULL;
	  JC_DV_DsetMenu = JC_Menu_Build(rowColumn1, XmMENU_OPTION, "in Dataset:", NULL, FALSE, JC_DV_menu_datasets);
	  XtManageChild(UxGetWidget(JC_DV_DsetMenu));
     }

/* Set values */
     strcpy(DV_ptr->var[0], var);
     strcpy(DV_ptr->dset[0], dset);
     strcpy(DV_ptr->assigned_dset, O_ptr->variable.dset);
     DV_ptr->clone_ptr[0] = O_ptr;
     JC_LetCommand_CreateClonedVarExpression(var_expression, 0, DV_ptr);
     XmTextSetString(textField3_EXP, var_expression);
     XtVaSetValues(UxGetWidget(label_dset1_EXP),
		   RES_CONVERT(XmNlabelString,  dset),
		   NULL);

/* Free allocated memory */
     XtFree(var);
     XtFree(dset);
     XmStringFree(label);
}


