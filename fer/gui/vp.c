static void InitPixmaps()
{
	XtVaSetValues(UxGetWidget(toggleButton62),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_corners_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_corners_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_corners_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton63),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_lr_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_lr_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_lr_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton64),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_lr_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_lr_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_lr_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton37),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpfull_xpm),
		XmNselectPixmap, GetPixmapFromData(vpfull_sel_xpm),
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton38),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpll_xpm),
		XmNselectPixmap, GetPixmapFromData(vpll_sel_xpm),
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton39),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vplr_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vplr_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);  

	XtVaSetValues(UxGetWidget(toggleButton40),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpur_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpur_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton41),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpll_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpll_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton42),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpleft_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpleft_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton43),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpright_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpright_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton44),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpupper_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpupper_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton45),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vplower_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vplower_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);
}

static void ClearBtnCB()
{
	ferret_command("SET WINDOW/CLEAR", IGNORE_COMMAND_WIDGET);
	/*gSomethingIsPlotted = 0;
	MaintainMainWdBtns(); */
}

static void InitInterface()
{
	/* initialize the viewports */
	ClearAllViewPorts();

	/* cycle buttons insensitive */
	ClearCycleButtons();

	if (gViewportActive) {
		switch (gCurrViewportType) {
			case 0:
				switch (gCurrViewportCycle) {
					case 0:
						XtVaSetValues(UxGetWidget(toggleButton38),
							XmNset, True,
							NULL);
						break;
					case 1:
						XtVaSetValues(UxGetWidget(toggleButton39),
							XmNset, True,
							NULL);
						break;
					case 2:
						XtVaSetValues(UxGetWidget(toggleButton40),
							XmNset, True,
							NULL);
						break;
					case 3:
						XtVaSetValues(UxGetWidget(toggleButton41),
							XmNset, True,
							NULL);
						break;
				}
				if (gViewportIsCycling) {
					XtVaSetValues(UxGetWidget(toggleButton62),
						XmNset, True,
						NULL);
				}
				break;
			case 1:
				switch (gCurrViewportCycle) {
					case 0:
						XtVaSetValues(UxGetWidget(toggleButton42),
							XmNset, True,
							NULL);
						break;
					case 1:
						XtVaSetValues(UxGetWidget(toggleButton43),
							XmNset, True,
							NULL);
						break;
				}
				if (gViewportIsCycling) {
					XtVaSetValues(UxGetWidget(toggleButton63),
						XmNset, True,
						NULL);
				}
				break;
			case 2:
				switch (gCurrViewportCycle) {
					case 0:
						XtVaSetValues(UxGetWidget(toggleButton44),
							XmNset, True,
							NULL);
						break;
						break;
					case 1:
						XtVaSetValues(UxGetWidget(toggleButton45),
							XmNset, True,
							NULL);
						break;
						break;
				}
				if (gViewportIsCycling) {
					XtVaSetValues(UxGetWidget(toggleButton64),
						XmNset, True,
						NULL);
				}
				break;
		}
	}
	else {
		/* full window selected */
		XtVaSetValues(UxGetWidget(toggleButton37),
			XmNset, True,
			NULL);
	}
}

static void ClearCycleButtons()
{
	XtVaSetValues(UxGetWidget(toggleButton62),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton63),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton64),
		XmNset, False,
		NULL);
}

static void ClearAllViewPorts()
{
	XtVaSetValues(UxGetWidget(toggleButton37),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton38),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton39),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton40),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton41),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton42),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton43),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton44),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton45),
		XmNset, False,
		NULL);
}
