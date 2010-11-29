/**
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
 */
package gov.noaa.pmel.ferret.threddsBrowser;

import java.awt.Component;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.util.Vector;

import javax.swing.Box;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSeparator;
import javax.swing.JTextField;
import javax.swing.WindowConstants;

/**
 * Dialog for selecting a location to display in a ThreddsBrowser.
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class LocationSelectorDialog extends JDialog implements ActionListener, FocusListener {
	private static final long serialVersionUID = 5072476287316578226L;

	/** Radio button for using the saved locations combo box */
	private JRadioButton savedLocsButton;
	/** Combo box of saved locations */
	private JComboBox savedLocsComboBox;
	/** Radio button for using the new location text field */
	private JRadioButton newLocButton;
	/** Text field for entering a new location */
	private JTextField newLocTextField;
	/** Radio button for selecting the root of a local directory tree */
	private JRadioButton localDirTreeButton;
	/** Select button to dismiss the dialog, using the selected location */
	private JButton selectButton;
	/** Cancel button to dismiss the dialog without selecting a location */
	private JButton cancelButton;
	/** Last radio button selected */
	private JRadioButton lastSelectedRadioButton;
	/** Default for the local directory file selector */
	private File localBrowseDir;

	/**
	 * Creates, but does not bring up, a LocationSelectorDialog with the given parameters.
	 * Use the selectLocation method to bring up the dialog and retrieve the location to use.
	 * @param parent create the dialog in the Frame for this component; can be null.
	 * @param savedLocations use these locations for the drop-down list of saved locations; 
	 * can be null or empty.
	 * @param localBrowseDir use this directory as the default root of the local 
	 * directory tree; can be null.
	 */
	public LocationSelectorDialog(Component parent, Vector<String> savedLocations, File localBrowseDir) {
		super(JOptionPane.getFrameForComponent(parent), "Select a Location to Display", true);
		this.localBrowseDir = localBrowseDir;

		setLayout(new GridBagLayout());
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.anchor = GridBagConstraints.NORTHWEST;

		// Label for the radio buttons
		gbc.gridx = 0;     gbc.gridy = 0;
		gbc.gridwidth = 2; gbc.gridheight = 1;
		gbc.weightx = 0.0; gbc.weighty = 0.0;
		gbc.fill = GridBagConstraints.NONE;
		gbc.insets = new Insets(10,10,10,10);
		add(new JLabel("Select a location to display in the browser:"), gbc);

		// Radio button for the drop-down list of saved locations
		savedLocsButton = new JRadioButton("Saved locations:");
	    savedLocsButton.setFocusPainted(false);
	    savedLocsButton.addActionListener(this);
 		gbc.gridx = 0;     gbc.gridy = 1;
		gbc.gridwidth = 1; gbc.gridheight = 1;
		gbc.weightx = 0.0; gbc.weighty = 1.0;
		gbc.fill = GridBagConstraints.VERTICAL;
		gbc.insets = new Insets(6,10,6,5);
		add(savedLocsButton, gbc);

		// JComboBox of the saved locations
		if ( savedLocations != null )
			savedLocsComboBox = new JComboBox(savedLocations);
		else
			savedLocsComboBox = new JComboBox();
		savedLocsComboBox.setEditable(false);
		savedLocsComboBox.addActionListener(this);
		savedLocsComboBox.addFocusListener(this);
		gbc.gridx = 1;     gbc.gridy = 1;
		gbc.gridwidth = 1; gbc.gridheight = 1;
		gbc.weightx = 1.0; gbc.weighty = 1.0;
		gbc.fill = GridBagConstraints.BOTH;
		gbc.insets = new Insets(5,5,5,10);
		add(savedLocsComboBox, gbc);

		// Radio button for entering a new location in a TextField
		newLocButton = new JRadioButton("New location:");
	    newLocButton.setFocusPainted(false);
	    newLocButton.addActionListener(this);
		gbc.gridx = 0;     gbc.gridy = 2;
		gbc.gridwidth = 1; gbc.gridheight = 1;
		gbc.weightx = 0.0; gbc.weighty = 1.0;
		gbc.fill = GridBagConstraints.VERTICAL;
		gbc.insets = new Insets(5,10,5,5);
		add(newLocButton, gbc);

		// JTextField to enter a new location
		newLocTextField = new JTextField(40);
		newLocTextField.addFocusListener(this);
		gbc.gridx = 1;     gbc.gridy = 2;
		gbc.gridwidth = 1; gbc.gridheight = 1;
		gbc.weightx = 1.0; gbc.weighty = 1.0;
		gbc.fill = GridBagConstraints.BOTH;
		gbc.insets = new Insets(5,5,5,10);
		add(newLocTextField, gbc);

		// Radio button to select the root directory of a local directory tree
		localDirTreeButton = new JRadioButton("Find a local directory to be the root of a tree of datasets ...");
	    localDirTreeButton.setFocusPainted(false);
	    localDirTreeButton.addActionListener(this);
		gbc.gridx = 0;     gbc.gridy = 3;
		gbc.gridwidth = 2; gbc.gridheight = 1;
		gbc.weightx = 1.0; gbc.weighty = 1.0;
		gbc.fill = GridBagConstraints.BOTH;
		gbc.insets = new Insets(5,10,5,10);
		add(localDirTreeButton, gbc);

		// Group these radio buttons
		ButtonGroup radioGroup = new ButtonGroup();
		radioGroup.add(savedLocsButton);
		radioGroup.add(newLocButton);
		radioGroup.add(localDirTreeButton);

		// Separator between the radio buttons and the select/cancel buttons
		gbc.gridx = 0;     gbc.gridy = 4;
		gbc.gridwidth = 2; gbc.gridheight = 1;
		gbc.weightx = 1.0; gbc.weighty = 0.0;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.insets = new Insets(5,0,5,0);
		add(new JSeparator(), gbc);

		// Create a horizontal panel with Select and Cancel buttons equally spaced
		JPanel horizPanel = new JPanel(new GridLayout(1,0));
        // Horizontal space
        horizPanel.add(Box.createHorizontalGlue());
        // Select button
        selectButton = new JButton("Select");
        selectButton.setFocusPainted(false);
        selectButton.addActionListener(this);
        horizPanel.add(selectButton);
        // Horizontal space
        horizPanel.add(Box.createHorizontalGlue());
        // Cancel button
        cancelButton = new JButton("Cancel");
        cancelButton.setFocusPainted(false);
        cancelButton.addActionListener(this);
        horizPanel.add(cancelButton);
        // Horizontal space
        horizPanel.add(Box.createHorizontalGlue());

        // Add the horizontal panel
		gbc.gridx = 0;     gbc.gridy = 5;
		gbc.gridwidth = 2; gbc.gridheight = 1;
		gbc.weightx = 1.0; gbc.weighty = 0.0;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.insets = new Insets(5,10,10,10);
        add(horizPanel, gbc);

       // Final touches to the dialog
        getRootPane().setDefaultButton(selectButton);
        pack();
        setLocationRelativeTo(parent);

        // Treat the dialog close ("X") button the same as the cancel button
        setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent evt) {
				cancelButton.doClick();
			}
		});

	}

	/**
	 * Display this LocationSelectorDialog and retrieve the location to use.
	 * @return the selected location as a trimmed, non-empty String, 
	 * or null if the user canceled the dialog.
	 */
	public String selectLocation() {
		// Set the saved locations radio button as the default
		savedLocsButton.doClick();
		
		// Because this dialog is modal, the following will block 
		// (but still allow event processing) until this dialog is dismissed
		setVisible(true);

		// Get the value associated with the last selected radio button
		String selectedLocation = null;
		if ( lastSelectedRadioButton == savedLocsButton ) {
			// Get the location from the saved locations combo box
			selectedLocation = (String) savedLocsComboBox.getSelectedItem();
		}
		else if ( lastSelectedRadioButton == newLocButton ) {
			// Get the location from the new location text field
			selectedLocation = newLocTextField.getText();
		}
		else if ( lastSelectedRadioButton == localDirTreeButton ) {
			// Create a file chooser opened to the default local directory
			JFileChooser chooser = new JFileChooser(localBrowseDir);

			// Customize the file chooser for directories
			chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
			chooser.setDialogTitle("Directory to be Root of a Local Dataset Tree");
			chooser.setApproveButtonToolTipText("Use the selected directory as the root of a local dataset tree");

			// Ugly hack to rename the label "File Name:" to "Selected Dir:" and 
			// to get rid of the file filter panel with the label "Files of Type:".
			// This will fail in locales that do not use exactly these strings
			// or if these components are not nested in JPanels.
			// More stuff might be removed if the file filter label and combo box 
			// are not in their own JPanel.
			for ( Component comp : chooser.getComponents() ) {
				checkComponentForFileLabel(comp);
			}

			// Get the selected directory
			if ( chooser.showDialog(this, "Use Selected Dir") == JFileChooser.APPROVE_OPTION ) {
				localBrowseDir = chooser.getSelectedFile();
				selectedLocation = localBrowseDir.getPath();
			}
		}

		// Trim the location; if blank, set to null
		if ( selectedLocation != null ) {
			selectedLocation = selectedLocation.trim();
			if ( selectedLocation.isEmpty() )
				selectedLocation = null;
		}

		// Return the selected location, which will be null if a dialog was canceled
		return selectedLocation;
	}

	/**
	 * If comp is a JLabel with text "File Name:", change the text to "Selected Dir:";
	 * if comp is a JLabel with text "Files of Type:", remove the JPanel that contains
	 * this component from it's JPanel; otherwise, if comp is a JPanel, call this 
	 * function on each of its components.
	 */
	private int checkComponentForFileLabel(Component comp) {
		if ( JLabel.class.isInstance(comp) ) {
			JLabel label = (JLabel) comp;
			if ( "File Name:".equals(label.getText()) ) {
				// Rename this label and continue on
				label.setText("Selected Dir:");
			}
			else if ( "Files of Type:".equals(label.getText()) ) {
				// This is the file filter label
				return 1;
			}
		}
		else if ( JPanel.class.isInstance(comp) ) {
			JPanel panel = (JPanel) comp;
			for ( Component panelComp : panel.getComponents() ) {
				int level = checkComponentForFileLabel(panelComp);
				if ( level == 2 ) {
					// Remove this panel component (which contains the file filter label) from this panel
					panel.remove(panelComp);
					return 0;
				}
				if ( level == 1 ) {
					// This is the panel containing the file filter label
					return 2;
				}
			}
		}
		return 0;
	}

	/**
	 * If from a radio button, records this currently selected radio button.
	 * If from a component associated with a radio button, makes sure the associated
	 * radio button is selected.
	 * If from the select button, closes this dialog.
	 * If from the cancel button, sets the currently selected radio button to null 
	 * and closes this dialog. 
	 */
	@Override
	public void actionPerformed(ActionEvent evt) {
		Object src = evt.getSource();
		if ( src == savedLocsButton ) {
			// Record that the saved locations radio button was selected
			lastSelectedRadioButton = savedLocsButton;
		}
		else if ( src == newLocButton ) {
			// Record that the new location radio button was selected
			lastSelectedRadioButton = newLocButton;
		}
		else if ( src == localDirTreeButton ) {
			// Record that the local directory tree radio button was selected
			lastSelectedRadioButton = localDirTreeButton;
		}
		else if ( src == savedLocsComboBox ) {
			// Put the selected location into the new location text field
			String location = (String) savedLocsComboBox.getSelectedItem();
			newLocTextField.setText(location);
			// Make sure the saved locations radio button is selected
			if ( lastSelectedRadioButton != savedLocsButton )
				savedLocsButton.doClick();
		}
		else if ( src == selectButton ) {
			// Close the dialog so selectLocation can proceed,
			// leaving lastActionCommand as-is to indicate how to proceed.
			setVisible(false);
		}
		else if ( src == cancelButton ) {
			// Set lastSelectedRadioButton to indicate the dialog was canceled
			// and close the dialog so selectLocation can proceed.
			lastSelectedRadioButton = null;
			setVisible(false);
		}
		else {
			System.out.println("unhandled action event: " + evt);
			System.out.println("coming from: " + src);
		}
	}

	/**
	 * Makes sure the radio button associated with the component gaining focus is selected.
	 */
	@Override
	public void focusGained(FocusEvent evt) {
		Object src = evt.getSource();
		if ( src == newLocTextField ) {
			// Make sure the new location radio button is selected
			if ( lastSelectedRadioButton != newLocButton )
				newLocButton.doClick();
		}
		else if ( src == savedLocsComboBox ) {
			// Make sure the saved locations radio button is selected
			if ( lastSelectedRadioButton != savedLocsButton )
				savedLocsButton.doClick();
		}
		else {
			System.out.println("unhandled focusGained event: " + evt);
			System.out.println("coming from: " + src);
		}
	}

	/**
	 * Does nothing.
	 */
	@Override
	public void focusLost(FocusEvent evt) {
		; // don't care
	}

}
