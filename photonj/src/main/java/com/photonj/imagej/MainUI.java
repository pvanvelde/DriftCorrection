package com.photonj.imagej;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Menu;
import java.awt.MenuItem;
import java.awt.PopupMenu;
import java.awt.Rectangle;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import org.scijava.app.StatusService;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import org.scijava.ui.UserInterface;

import com.photonj.binding.Binding;
import com.photonj.binding.Binding.ICameraCalibration;
import com.photonj.binding.Binding.ISpotDetector;
import com.photonj.binding.Binding.NativeAPI;
import com.photonj.binding.Binding.NativeAPI.SpotDetectorConfig;
import com.photonj.binding.Binding.Spot;
import com.photonj.binding.Binding.SpotDetectParams;
import com.photonj.binding.Vector3;
import com.photonj.binding.ZStack;
import com.photonj.imagej.LocRenderWindow.View;
import com.sun.jna.Pointer;



import ij.IJ;
import ij.IJEventListener;
import ij.ImageListener;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.WindowManager;
import ij.gui.ImageCanvas;
import ij.gui.Line;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.process.FloatPolygon;
import ij.process.FloatProcessor;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.stats.Max;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;

import javax.swing.JTextField;
import javax.swing.JSlider;
import javax.swing.JLabel;
import java.awt.event.ActionListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.event.WindowStateListener;
import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.security.CodeSource;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.awt.event.ActionEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.SpinnerNumberModel;
import javax.swing.JRadioButton;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JTextPane;
import java.awt.SystemColor;
import java.awt.Window;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import java.awt.Dialog.ModalityType;
import java.awt.FileDialog;

import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.JMenuBar;
import javax.swing.JMenu;

public class MainUI extends JDialog  implements LocRenderWindow.ILocInfoDisplay {

	private final JPanel contentPanel = new JPanel();

	volatile Thread processingThread;
	
	private UIService ui;
	
	Timer renderWindowUpdateTimer;
	
	JComponent[] componentsWithPersistedState;
	
	private Binding cpp;
	
	private JSpinner spinnerROISize;
	private JSpinner spinnerCameraGain;
	private JSpinner spinnerCameraOffset;
	
	ImagePlus currentImage;
	int currentSlice;
	private JSpinner spinnerSigmaX;
	private JSpinner spinnerSigmaY;
	
	private JSpinner spinnerSRSigma;
	private JRadioButton radioAvgCRLB;
	private JRadioButton radioFixedWidth;
	
	String datasetName;

	private JSpinner spinnerZoom;
	private JTextField textDLLPath;
	private JTextField textCameraOffsetFile;
	private JTextField textCameraGainFile;
	private final ButtonGroup buttonGroup = new ButtonGroup();
	private JLabel lblInfo;
	private JTextField textCubicSplinePSF;
	private final ButtonGroup buttonGroupPSFType = new ButtonGroup();
	private JCheckBox checkEstimateSigmaXY;
	private JRadioButton radioPSFTypeGauss2D;
	private JRadioButton radioPSFTypeSpline;
	private JTextField textBackgroundImage;

	private JSpinner spinnerPSFCorrZPlanes;
	private JSpinner spinnerBackgroundFilter;
	private JSpinner spinnerMaxDistanceXY;
	private JSpinner spinnerDetectionThreshold;
	private JTextField textField;
	private JSpinner spinnerPixelSizeX;
	private JSpinner spinnerPixelSizeY;
	
	/**
	 * Create the dialog.
	 */
	public MainUI() {
		super(null, ModalityType.MODELESS);
		setTitle("PhotonJ - 3D Localization microscopy");
		
		System.out.println( "PhotonJ init");
		
		//setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		setBounds(100, 100, 536, 390);
		getContentPane().setLayout(new BorderLayout());
		contentPanel.setBorder(new EmptyBorder(5, 5, 5, 5));
		getContentPane().add(contentPanel, BorderLayout.CENTER);
		
		JTabbedPane tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		
		JPanel panel = new JPanel();
		tabbedPane.addTab("PSF", null, panel, null);
		panel.setLayout(null);
		
		radioPSFTypeGauss2D = new JRadioButton("2D Gaussian");
		buttonGroupPSFType.add(radioPSFTypeGauss2D);
		radioPSFTypeGauss2D.setSelected(true);
		radioPSFTypeGauss2D.setBounds(6, 7, 109, 23);
		panel.add(radioPSFTypeGauss2D);
		
		radioPSFTypeSpline = new JRadioButton("Cubic Spline");
		radioPSFTypeSpline.setName("psf_spline");
		buttonGroupPSFType.add(radioPSFTypeSpline);
		radioPSFTypeSpline.setBounds(6, 103, 109, 23);
		panel.add(radioPSFTypeSpline);
		
		spinnerSigmaX = new JSpinner();
		spinnerSigmaX.setName("psf_sigmaX");
		spinnerSigmaX.setModel(new SpinnerNumberModel(new Float(1.5), new Float(1), null, new Float(0.05)));
		spinnerSigmaX.setBounds(54, 36, 61, 20);
		panel.add(spinnerSigmaX);
		
		spinnerSigmaY = new JSpinner();
		spinnerSigmaY.setName("psf_sigmaY");
		spinnerSigmaY.setModel(new SpinnerNumberModel(new Float(1.5), new Float(1), null, new Float(0.05)));
		spinnerSigmaY.setBounds(54, 67, 61, 20);
		panel.add(spinnerSigmaY);
		
		JLabel lblSigmaX = new JLabel("Sigma X");
		lblSigmaX.setBounds(125, 39, 91, 14);
		panel.add(lblSigmaX);
		
		JLabel lblSigmaY = new JLabel("Sigma Y");
		lblSigmaY.setBounds(125, 70, 91, 14);
		panel.add(lblSigmaY);
		
		checkEstimateSigmaXY = new JCheckBox("Estimate sigma X,Y");
		checkEstimateSigmaXY.setName("psf_estimate_sigma");
		checkEstimateSigmaXY.setBounds(116, 7, 145, 23);
		panel.add(checkEstimateSigmaXY);
		
		JButton btnCSplineBrowsePSF = new JButton("Browse");
		btnCSplineBrowsePSF.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				browseForPSF();
			}
		});
		btnCSplineBrowsePSF.setBounds(367, 161, 118, 23);
		panel.add(btnCSplineBrowsePSF);
		
		textCubicSplinePSF = new JTextField();
		textCubicSplinePSF.setName("cspline_file");
		textCubicSplinePSF.setBounds(27, 162, 330, 20);
		panel.add(textCubicSplinePSF);
		textCubicSplinePSF.setColumns(10);
		
		JButton btnCreatePSF = new JButton("Create PSF from Z stacks");
		btnCreatePSF.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				CreatePSFDialog dlg = new CreatePSFDialog();
				dlg.setVisible(true);
			}
		});
		btnCreatePSF.setBounds(153, 189, 204, 23);
		panel.add(btnCreatePSF);
		
		JLabel lblPsfForLocalization = new JLabel("PSF for localization:");
		lblPsfForLocalization.setBounds(27, 137, 153, 14);
		panel.add(lblPsfForLocalization);
		
		JButton btnViewPSF = new JButton("View PSF");
		btnViewPSF.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				viewPSF();
			}
		});
		btnViewPSF.setBounds(26, 189, 117, 23);
		panel.add(btnViewPSF);
		
		JPanel panelSpotDetect = new JPanel();
		tabbedPane.addTab("Spot detection", null, panelSpotDetect, null);
		panelSpotDetect.setLayout(null);
		
		JLabel lblDetectionThreshold = new JLabel("Detection threshold (Arbitrary unit):");
		lblDetectionThreshold.setBounds(11, 46, 204, 14);
		panelSpotDetect.add(lblDetectionThreshold);
		
		textBackgroundImage = new JTextField();
		textBackgroundImage.setName("sd_bg_image");
		textBackgroundImage.setBounds(165, 76, 294, 20);
		panelSpotDetect.add(textBackgroundImage);
		textBackgroundImage.setColumns(10);
		
		JButton btnBrowseSpotDetectioBgImage = new JButton("Browse");
		btnBrowseSpotDetectioBgImage.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		btnBrowseSpotDetectioBgImage.setBounds(166, 103, 89, 23);
		panelSpotDetect.add(btnBrowseSpotDetectioBgImage);
		
		JLabel lblBackgroundImage = new JLabel("Background image");
		lblBackgroundImage.setBounds(10, 79, 152, 14);
		panelSpotDetect.add(lblBackgroundImage);
		
		JLabel lblRoiSize = new JLabel("ROI Size:");
		lblRoiSize.setBounds(10, 14, 92, 14);
		panelSpotDetect.add(lblRoiSize);
		
		spinnerROISize = new JSpinner();
		spinnerROISize.setName("roisize");
		spinnerROISize.setBounds(166, 11, 109, 20);
		panelSpotDetect.add(spinnerROISize);
		spinnerROISize.setModel(new SpinnerNumberModel(new Integer(10), new Integer(2), null, new Integer(1)));
		
		spinnerPSFCorrZPlanes = new JSpinner();
		spinnerPSFCorrZPlanes.setName("sd_psfcorr_zplanes");
		spinnerPSFCorrZPlanes.setModel(new SpinnerNumberModel(new Integer(1), new Integer(1), null, new Integer(1)));
		spinnerPSFCorrZPlanes.setBounds(395, 226, 64, 20);
		panelSpotDetect.add(spinnerPSFCorrZPlanes);
		
		JLabel lblNewLabel_1 = new JLabel("Number of Z planes used for detection (for cubic spline PSF):");
		lblNewLabel_1.setHorizontalAlignment(SwingConstants.RIGHT);
		lblNewLabel_1.setBounds(33, 229, 352, 14);
		panelSpotDetect.add(lblNewLabel_1);
		
		JTextPane txtpnSpotDetectionIs = new JTextPane();
		txtpnSpotDetectionIs.setBackground(SystemColor.control);
		txtpnSpotDetectionIs.setText("Spot detection is done by detecting peaks in the correlation with PSF slices.");
		txtpnSpotDetectionIs.setBounds(11, 136, 396, 34);
		panelSpotDetect.add(txtpnSpotDetectionIs);
		
		JLabel lblGaussianFilterSize = new JLabel("Uniform filter size for background estimation [pixels]:");
		lblGaussianFilterSize.setHorizontalAlignment(SwingConstants.TRAILING);
		lblGaussianFilterSize.setBounds(53, 204, 332, 14);
		panelSpotDetect.add(lblGaussianFilterSize);
		
		spinnerBackgroundFilter = new JSpinner();
		spinnerBackgroundFilter.setModel(new SpinnerNumberModel(new Integer(1), new Integer(1), null, new Integer(1)));
		spinnerBackgroundFilter.setName("sd_bg_filter_size");
		spinnerBackgroundFilter.setBounds(395, 202, 64, 20);
		panelSpotDetect.add(spinnerBackgroundFilter);
		
		spinnerMaxDistanceXY = new JSpinner();
		spinnerMaxDistanceXY.setModel(new SpinnerNumberModel(new Integer(10), new Integer(1), null, new Integer(1)));
		spinnerMaxDistanceXY.setToolTipText("Distance is based on uniform norm, not euclidian (distance must be further than this value on all axes)");
		spinnerMaxDistanceXY.setName("sd_max_distance");
		spinnerMaxDistanceXY.setBounds(394, 179, 64, 20);
		panelSpotDetect.add(spinnerMaxDistanceXY);
		
		JLabel lblMinimumDistancepixels = new JLabel("Minimum distance between spot centers in X and Y [pixels]:");
		lblMinimumDistancepixels.setHorizontalAlignment(SwingConstants.TRAILING);
		lblMinimumDistancepixels.setBounds(54, 182, 332, 14);
		panelSpotDetect.add(lblMinimumDistancepixels);
		
		spinnerDetectionThreshold = new JSpinner();
		spinnerDetectionThreshold.setModel(new SpinnerNumberModel(new Float(0), new Float(0), null, new Float(1)));
		spinnerDetectionThreshold.setToolTipText("Distance is based on uniform norm, not euclidian (distance must be further than this value on all axes)");
		spinnerDetectionThreshold.setName("sd_threshold");
		spinnerDetectionThreshold.setBounds(395, 43, 64, 20);
		panelSpotDetect.add(spinnerDetectionThreshold);
		
		JPanel panelCameraCalib = new JPanel();
		tabbedPane.addTab("Camera calibration", null, panelCameraCalib, null);
		panelCameraCalib.setLayout(null);
		
		JLabel lblCameraGain = new JLabel("Camera gain");
		lblCameraGain.setBounds(10, 14, 92, 14);
		panelCameraCalib.add(lblCameraGain);
		
		JLabel lblCameraOffset = new JLabel("Camera offset");
		lblCameraOffset.setBounds(10, 39, 92, 14);
		panelCameraCalib.add(lblCameraOffset);
		
		spinnerCameraGain = new JSpinner();
		spinnerCameraGain.setName("camera_gain");
		spinnerCameraGain.setModel(new SpinnerNumberModel(new Float(1), new Float(0), null, new Float(1)));
		spinnerCameraGain.setBounds(199, 14, 109, 20);
		panelCameraCalib.add(spinnerCameraGain);
		
		spinnerCameraOffset = new JSpinner();
		spinnerCameraOffset.setName("camera_offset");
		spinnerCameraOffset.setModel(new SpinnerNumberModel(new Float(0), new Float(0), null, new Float(1)));
		spinnerCameraOffset.setBounds(199, 39, 109, 20);
		panelCameraCalib.add(spinnerCameraOffset);
		
		textCameraOffsetFile = new JTextField();
		textCameraOffsetFile.setName("camera_offset_file");
		textCameraOffsetFile.setBounds(183, 157, 191, 20);
		panelCameraCalib.add(textCameraOffsetFile);
		textCameraOffsetFile.setColumns(10);
		
		JCheckBox chckbxCameraCalibPerPixel = new JCheckBox("Use sCMOS per-pixel gain/offsets/variance");
		chckbxCameraCalibPerPixel.setName("use_per_pixel_camera_calib");
		chckbxCameraCalibPerPixel.setBounds(10, 127, 310, 23);
		panelCameraCalib.add(chckbxCameraCalibPerPixel);
		
		JLabel lblCameraDarkFrames = new JLabel("Camera offset:");
		lblCameraDarkFrames.setBounds(10, 160, 191, 14);
		panelCameraCalib.add(lblCameraDarkFrames);
		
		JLabel lblCameraLightFrames = new JLabel("Camera gain file:");
		lblCameraLightFrames.setBounds(10, 192, 155, 14);
		panelCameraCalib.add(lblCameraLightFrames);
		
		textCameraGainFile = new JTextField();
		textCameraGainFile.setName("camera_gain_file");
		textCameraGainFile.setColumns(10);
		textCameraGainFile.setBounds(183, 189, 191, 20);
		panelCameraCalib.add(textCameraGainFile);
		
		JButton btnCameraOffsetBrowse = new JButton("Browse");
		btnCameraOffsetBrowse.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		btnCameraOffsetBrowse.setBounds(384, 156, 101, 23);
		panelCameraCalib.add(btnCameraOffsetBrowse);
		
		JButton buttonCameraGainBrowse = new JButton("Browse");
		buttonCameraGainBrowse.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		buttonCameraGainBrowse.setBounds(384, 188, 101, 23);
		panelCameraCalib.add(buttonCameraGainBrowse);
		
		JLabel lblCameraVariance = new JLabel("Camera readnoise variance:");
		lblCameraVariance.setBounds(10, 223, 155, 14);
		panelCameraCalib.add(lblCameraVariance);
		
		textField = new JTextField();
		textField.setName("camera_readnoise_file");
		textField.setColumns(10);
		textField.setBounds(183, 221, 191, 20);
		panelCameraCalib.add(textField);
		
		JButton buttonCamReadnoiseBrowse = new JButton("Browse");
		buttonCamReadnoiseBrowse.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		buttonCamReadnoiseBrowse.setBounds(384, 220, 101, 23);
		panelCameraCalib.add(buttonCamReadnoiseBrowse);
		
		JLabel lblNewLabel = new JLabel("Pixel size (X) [nm]");
		lblNewLabel.setBounds(10, 66, 127, 14);
		panelCameraCalib.add(lblNewLabel);
		
		JLabel lblPixelSizey = new JLabel("Pixel size (Y) [nm]");
		lblPixelSizey.setBounds(10, 92, 127, 14);
		panelCameraCalib.add(lblPixelSizey);
		
		spinnerPixelSizeX = new JSpinner();
		spinnerPixelSizeX.setModel(new SpinnerNumberModel(new Integer(100), new Integer(1), null, new Integer(1)));
		spinnerPixelSizeX.setName("pixelsize_x");
		spinnerPixelSizeX.setBounds(199, 63, 109, 20);
		panelCameraCalib.add(spinnerPixelSizeX);
		
		spinnerPixelSizeY = new JSpinner();
		spinnerPixelSizeY.setModel(new SpinnerNumberModel(new Integer(100), new Integer(1), null, new Integer(1)));
		spinnerPixelSizeY.setName("pixelsize_y");
		spinnerPixelSizeY.setBounds(199, 89, 109, 20);
		panelCameraCalib.add(spinnerPixelSizeY);
		
		JPanel panelRender = new JPanel();
		tabbedPane.addTab("Rendering", null, panelRender, null);
		panelRender.setLayout(null);

		
		ActionListener updateRenderActionListener = new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				updateRender();
			}
		};
		
		radioAvgCRLB = new JRadioButton("Average CRLB");
		radioAvgCRLB.addActionListener(updateRenderActionListener);
		radioAvgCRLB.setBounds(8, 66, 109, 23);
		panelRender.add(radioAvgCRLB);
		
		JLabel lblWidthOfRendered = new JLabel("Width of rendered Gaussian spots:");
		lblWidthOfRendered.setBounds(8, 45, 205, 14);
		panelRender.add(lblWidthOfRendered);
		
		radioFixedWidth = new JRadioButton("Fixed [sr pixels]:");
		radioFixedWidth.setSelected(true);
		radioFixedWidth.addActionListener(updateRenderActionListener);
		radioFixedWidth.setBounds(8, 92, 109, 23);
		panelRender.add(radioFixedWidth);
		
		spinnerSRSigma = new JSpinner();
		spinnerSRSigma.setModel(new SpinnerNumberModel(new Float(1), new Float(0), null, new Float(0.1)));
		spinnerSRSigma.setBounds(156, 92, 56, 20);
		panelRender.add(spinnerSRSigma);
		
		JLabel lblMeanCRLBX = new JLabel("");
		lblMeanCRLBX.setBounds(152, 35, 61, 14);
		panelRender.add(lblMeanCRLBX);
		
		spinnerZoom = new JSpinner();
		spinnerZoom.setModel(new SpinnerNumberModel(new Integer(8), new Integer(1), null, new Integer(1)));
		spinnerZoom.setBounds(159, 11, 56, 20);
		panelRender.add(spinnerZoom);
		
		JLabel lblSuperresolutionZoom = new JLabel("Super-resolution zoom:");
		lblSuperresolutionZoom.setBounds(10, 14, 139, 14);
		panelRender.add(lblSuperresolutionZoom);
		
		JButton btnLocalize = new JButton("Run Localization");
		btnLocalize.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				runLocalization();
			}
		});

		ButtonGroup btnGroupSRWidth=new ButtonGroup();
		btnGroupSRWidth.add(radioAvgCRLB);
		btnGroupSRWidth.add(radioFixedWidth);
		
		JRadioButton radioSpotCRLB = new JRadioButton("Individual spot CRLB");
		radioSpotCRLB.addActionListener(updateRenderActionListener);
		btnGroupSRWidth.add(radioSpotCRLB);
		radioSpotCRLB.setBounds(8, 118, 150, 23);
		panelRender.add(radioSpotCRLB);
		
		JLabel lblPropertyChangesAre = new JLabel("Changes are applied to the currently active image");
		lblPropertyChangesAre.setBounds(8, 157, 430, 23);
		panelRender.add(lblPropertyChangesAre);
		
		JPanel panel_1 = new JPanel();
		tabbedPane.addTab("DLL Path", null, panel_1, null);

    	panel_1.setLayout(null);
    	
    	JLabel lblSetEmptyTo = new JLabel("Set empty to use default DLL:");
    	lblSetEmptyTo.setBounds(20, 9, 386, 14);
    	panel_1.add(lblSetEmptyTo);
    	
    	JButton btnBrowseDLLPath = new JButton("Browse");
    	btnBrowseDLLPath.setBounds(406, 34, 79, 23);
    	btnBrowseDLLPath.addActionListener(new ActionListener() {
    		public void actionPerformed(ActionEvent e) {
    			OpenDialog od=new OpenDialog("Select photonpy DLL path:");
    			if (od.getPath() != null) {
	    			textDLLPath.setText(od.getPath());
	    			loadDLL();
    			}
    		}
    	});
    	
    	textDLLPath = new JTextField();
    	textDLLPath.setName("photonpy_dll_path");
    	textDLLPath.setBounds(20, 35, 371, 20);
    	textDLLPath.addActionListener(new ActionListener() {
    		public void actionPerformed(ActionEvent e) {
    			loadDLL();
    		}
    	});

    	panel_1.add(textDLLPath);
    	textDLLPath.setColumns(10);
    	
    	panel_1.add(btnBrowseDLLPath);
    	
    	lblInfo = new JLabel("");
    	
    	JButton btnUseActiveImg = new JButton("Use current image");
    	btnUseActiveImg.addActionListener(new ActionListener() {
    		public void actionPerformed(ActionEvent e) {
    			setActiveImage(IJ.getImage());
    		}
    	});
    	GroupLayout gl_contentPanel = new GroupLayout(contentPanel);
    	gl_contentPanel.setHorizontalGroup(
    		gl_contentPanel.createParallelGroup(Alignment.LEADING)
    			.addGroup(gl_contentPanel.createSequentialGroup()
    				.addContainerGap()
    				.addGroup(gl_contentPanel.createParallelGroup(Alignment.TRAILING)
    					.addGroup(gl_contentPanel.createSequentialGroup()
    						.addComponent(lblInfo, GroupLayout.DEFAULT_SIZE, 490, Short.MAX_VALUE)
    						.addContainerGap())
    					.addGroup(gl_contentPanel.createSequentialGroup()
    						.addComponent(btnUseActiveImg, GroupLayout.DEFAULT_SIZE, 140, Short.MAX_VALUE)
    						.addPreferredGap(ComponentPlacement.RELATED)
    						.addComponent(btnLocalize, GroupLayout.DEFAULT_SIZE, 152, Short.MAX_VALUE)
    						.addGap(202))
    					.addComponent(tabbedPane, GroupLayout.DEFAULT_SIZE, 500, Short.MAX_VALUE)))
    	);
    	gl_contentPanel.setVerticalGroup(
    		gl_contentPanel.createParallelGroup(Alignment.LEADING)
    			.addGroup(gl_contentPanel.createSequentialGroup()
    				.addGap(5)
    				.addGroup(gl_contentPanel.createParallelGroup(Alignment.BASELINE)
    					.addComponent(btnUseActiveImg)
    					.addComponent(btnLocalize))
    				.addPreferredGap(ComponentPlacement.RELATED)
    				.addComponent(tabbedPane, GroupLayout.DEFAULT_SIZE, 282, Short.MAX_VALUE)
    				.addPreferredGap(ComponentPlacement.RELATED)
    				.addComponent(lblInfo, GroupLayout.PREFERRED_SIZE, 14, GroupLayout.PREFERRED_SIZE)
    				.addGap(5))
    	);
    	contentPanel.setLayout(gl_contentPanel);
		
		ImagePlus.addImageListener(new ImageListener() {
			@Override
			public void imageUpdated(ImagePlus imp) {
				//System.out.println("Image slice: " + imp.getCurrentSlice());
				if (imp == currentImage && imp.getCurrentSlice() != currentSlice)
					updateROIOverlay();
			}
			
			@Override
			public void imageOpened(ImagePlus imp) {
			}
			
			@Override
			public void imageClosed(ImagePlus imp) {
				LocRenderWindow.imageClosed(imp);
				if(currentImage == imp)
					currentImage = null;
			}
		});
		
		ChangeListener updateRenderCL = new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				updateRender();
			}
		};
		ChangeListener updateROIOverlayCL = new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				updateROIOverlay();
			}
		};
		spinnerSigmaX.addChangeListener(updateROIOverlayCL);
		spinnerSigmaY.addChangeListener(updateROIOverlayCL);

		spinnerZoom.addChangeListener(updateRenderCL);
		spinnerSRSigma.addChangeListener(updateRenderCL);
		spinnerPSFCorrZPlanes.addChangeListener(updateROIOverlayCL);
		spinnerCameraOffset.addChangeListener(updateROIOverlayCL);
		spinnerBackgroundFilter.addChangeListener(updateROIOverlayCL);
		spinnerCameraGain.addChangeListener(updateROIOverlayCL);
		spinnerROISize.addChangeListener(updateROIOverlayCL);
		spinnerMaxDistanceXY.addChangeListener(updateROIOverlayCL);
		spinnerDetectionThreshold.addChangeListener(updateROIOverlayCL);

		MenuItem openROIMenu=new MenuItem("Render marked area (PhotonJ)");
		openROIMenu.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// Let me know if there's a better way of getting the image associated with this menu event...
				MenuItem item = (MenuItem)e.getSource();
				PopupMenu pm = (PopupMenu)item.getParent();
				ImageCanvas canvas = (ImageCanvas)pm.getParent();
				openMarkedROI(canvas);
			}
		});

		Menu photonJMenu = createPhotonJContextMenu();
		PopupMenu popup = ij.Menus.getPopupMenu();
		
		popup.insert(openROIMenu, 0);
		popup.insert(photonJMenu, 1);
		
		componentsWithPersistedState = new JComponent[] {
			spinnerPSFCorrZPlanes, chckbxCameraCalibPerPixel, radioPSFTypeSpline, spinnerCameraGain, spinnerCameraOffset, spinnerBackgroundFilter,
			spinnerROISize, spinnerSigmaX, spinnerSigmaY, textBackgroundImage, textCameraGainFile, textCameraOffsetFile, textCubicSplinePSF, textDLLPath,
			checkEstimateSigmaXY, spinnerDetectionThreshold, spinnerMaxDistanceXY, spinnerPixelSizeX, spinnerPixelSizeY
		};
		
		fixJSpinnerBug(componentsWithPersistedState);
		
		System.out.println("IJ Prefs dir: "+ Prefs.getPrefsDir());

		loadState();
    	loadDLL();

		ImagePlus img = WindowManager.getCurrentImage();
		if (img != null) {
			setActiveImage(img);
			updateROIOverlay();
		}

		IJ.getInstance().addWindowListener(new WindowListener() {
			
			@Override
			public void windowOpened(WindowEvent e) {
			}
			
			@Override
			public void windowIconified(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void windowDeiconified(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void windowDeactivated(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void windowClosing(WindowEvent e) {
				PersistUIState.storeState(componentsWithPersistedState);
			}
			
			@Override
			public void windowClosed(WindowEvent e) {
			}
			
			@Override
			public void windowActivated(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
		});
		
		
		this.addWindowListener(new WindowListener() {
			
			@Override
			public void windowOpened(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void windowIconified(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void windowDeiconified(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void windowDeactivated(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void windowClosing(WindowEvent e) {
				PersistUIState.storeState(componentsWithPersistedState);
				
				stopTimer();
			}
			
			@Override
			public void windowClosed(WindowEvent e) {
				// TODO Auto-generated method stub
		
			}
			
			@Override
			public void windowActivated(WindowEvent e) {
				// TODO Auto-generated method stub
				
			}
		});
		
		
		startTimer();
	}
	
	private Menu createPhotonJContextMenu() {
		Menu m = new Menu("PhotonJ");
		
		MenuItem setRes =new MenuItem("Change resolution");
		m.add(setRes);

		MenuItem export =new MenuItem("Export localizations to CSV/HDF5");
		m.add(export);

		MenuItem exportView =new MenuItem("Discard points outside view");
		m.add(exportView);

		MenuItem mergeData =new MenuItem("Merge dataset");
		m.add(mergeData);
		
		return m;
	}
	
	void stopTimer() { 
		if (renderWindowUpdateTimer != null) {
			renderWindowUpdateTimer.stop();
			renderWindowUpdateTimer = null;
		}
	}
	
	void startTimer() {
		renderWindowUpdateTimer = new Timer(50, new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// check if any render windows need updating based on ROI settings
				LocRenderWindow.timerUpdate();
			}
		});
		renderWindowUpdateTimer.start();
	}

	private void fixJSpinnerBug(JComponent[] list) {
		FocusListener l = new FocusListener() {
			@Override
			public void focusLost(FocusEvent e) {
				try {
					((JSpinner) e.getSource()).commitEdit();
				} catch (ParseException e1) {}
			}					
			@Override
			public void focusGained(FocusEvent e) {
				// TODO Auto-generated method stub
				
			}
		};
		
		for (JComponent c : list) {
			if (c instanceof JSpinner) {
				((JSpinner)c).addFocusListener(l);
			}
		}
	}

	void loadState() {
		PersistUIState.loadState(componentsWithPersistedState);
	}
	
	protected void setActiveImage(ImagePlus image) {

		if (image.getNChannels() != 1 || image.getBitDepth() != 16) {
			IJ.showMessage("Currently only single channel, 16 bit images are supported");
			currentImage = null;
			return;
		}
		
		currentImage = image;
		updateROIOverlay();
	}

	protected void browseForPSF() {
		String fn = FileDlg.saveDialog("Select ZStack file:", 
				"ZSTACK binary (*.zstack);zstack", 
				currentImage != null ? currentImage.getOriginalFileInfo().directory : null);
		
		if (fn != null) {
			textCubicSplinePSF.setText(fn);
		}
	}

	protected void openMarkedROI(ImageCanvas canvas) {
		ImagePlus im = canvas.getImage();
		LocRenderWindow wnd = LocRenderWindow.getWindow(im);
		
		if (wnd == null)
			return;
		
		wnd.renderMarkedROI();	
	}

	void loadDLL() {
    	if (cpp != null) {
    		cpp.Close();
    		cpp = null;
    	}
        CodeSource src=PhotonJ.class.getProtectionDomain().getCodeSource();
    	String path = src.getLocation().getPath();
    	System.out.println("PhotonJ Path: " + path);
    	String defaultPhotonpyPath = new File(path).getParent() + "/photonpy/win64/photonpy.dll";

    	String dllPath = defaultPhotonpyPath;
    	if (textDLLPath.getText().length() != 0)
    		dllPath = textDLLPath.getText();
    	
		if (!new File(dllPath).exists()) {
			IJ.showMessage("Photonpy dll path is invalid: " + dllPath);
		} else  {
    		cpp = new Binding(dllPath);
		}
	}
	
	void updateRender() {
		LocRenderWindow wnd=getActiveRenderWindow();

		if (wnd != null)
			updateRender(wnd);
	}
	
	void updateRender(LocRenderWindow wnd) {
		if (cpp == null)
			return;
		
		int zoom = (int)spinnerZoom.getValue();
		float sigma;
		
		if (radioAvgCRLB.isSelected())
			sigma = wnd.getResults().getMeanCRLBX();
		else if(radioFixedWidth.isSelected())
			sigma = ((float)spinnerSRSigma.getValue())/zoom;
		else 
			sigma = -1.0f;
		
		int w = wnd.data.originalWidth*zoom;
		int h = wnd.data.originalHeight*zoom;
		
		Transform tr=new Transform();
		tr.m[0].x*=zoom/wnd.data.scaling.x;
		tr.m[1].y*=zoom/wnd.data.scaling.y;
		
		wnd.view.sigma = sigma/zoom;
		wnd.view.transform=tr;
		wnd.update(wnd.view);
	}
	

	void updateROIOverlay() {
		
		if (currentImage == null)
			return;
		ImagePlus img = currentImage;
				
		currentSlice = img.getCurrentSlice();
		
    	Overlay overlay = img.getOverlay();
    	if (overlay == null) {
    		overlay = new Overlay();
    		img.setOverlay(overlay);
    	}
		overlay.clear();

		if (cpp != null) {
	    	Binding.ICameraCalibration cameraCalib = getCameraCalibration();
			Localizer loc = createLocalizer(img.getWidth(), img.getHeight());
	    	Binding.ROI[] rois = loc.detectSpots(cameraCalib, img, currentSlice);
	    	loc.close();
	    	
	    	cameraCalib.destroy();
	
			int roisize=getROISize();
			for (int i=0;i<rois.length;i++) {
				overlay.add(new Roi(rois[i].x, rois[i].y, roisize, roisize));
			}
		}
		currentImage.updateAndDraw();
	}
	
	
	private Localizer createLocalizer(int w, int h) {
		SpotDetectParams sdparams=new SpotDetectParams();
		
		sdparams.imageWidth = w;
		sdparams.imageHeight = h;
		sdparams.maxFilterSizeXY = 2*(int)spinnerMaxDistanceXY.getValue();
		sdparams.minThreshold=(float)spinnerDetectionThreshold.getValue();
		sdparams.bgUniformFilter = (int)spinnerBackgroundFilter.getValue();
		sdparams.debugMode=false;
				
		if (radioPSFTypeGauss2D.isSelected()) {
			boolean fitSigma = checkEstimateSigmaXY.isSelected();
			float sigmaX = (float)spinnerSigmaX.getValue();
			float sigmaY = (float)spinnerSigmaY.getValue();

			return Localizer.Gaussian2DPSFLocalizer(getROISize(), sigmaX, sigmaY, fitSigma, sdparams,  cpp);
		} else {
			int numZSlices = (int)spinnerPSFCorrZPlanes.getValue();
			return Localizer.CubicSplinePSFLocalizer(getROISize(), textCubicSplinePSF.getText(), numZSlices,sdparams,  cpp);
		}
	}

	Binding.ICameraCalibration getCameraCalibration() {
		float gain = Float.parseFloat(spinnerCameraGain.getValue().toString());
		float offset = Float.parseFloat(spinnerCameraOffset.getValue().toString());

		return cpp.createCameraCalibration(gain,offset);
	}

	
	void viewPSF() {
		String fn=textCubicSplinePSF.getText();
		ZStack psf;
		try {
			psf = ZStack.readFromFile(fn);
		} catch (IOException e) {
			IJ.showMessage("Failed to read " + fn);
			return;
		}
		
		ImagePlus img = IJ.createImage(fn, psf.shape[2], psf.shape[1], psf.shape[0], 32);
		
		for (int i=0;i<psf.shape[0];i++) {
			float[] pixels = psf.getSlice(i);
			
			float max=0.0f;
			for (int j=0;j<pixels.length;j++)
				if (max<pixels[j]) max=pixels[j];
			if (max>0.0f) {
				for (int j=0;j<pixels.length;j++)
					pixels[j] /= max;
			}
			img.getStack().setPixels(pixels, 1+i);
		}
		img.setDisplayRange(0.0f, 1.0f);
		ui.show(img);
	}
	
	float getDetectionThreshold() {
		return (float)spinnerDetectionThreshold.getValue();
	}
	
	int getROISize() {
		return Integer.parseInt(spinnerROISize.getValue().toString());
	}
	
	void exportLocalizations() {
		LocRenderWindow wnd = getActiveRenderWindow();
		
		if (wnd == null)
			return;
		
		LocResults results = wnd.getResults();

		// Query for filename to save data
		SaveDialog sd = new SaveDialog("Save HDF5 ...", "", ".hdf5");
		String directory = sd.getDirectory();
		String name = sd.getFileName();
		if (name == null || name.equals("")) {
			return;
		}
		String filename = directory + name;
		
		results.exportPicassoHDF5(filename);
	}
	
	LocRenderWindow getActiveRenderWindow() {
		ImagePlus img = WindowManager.getCurrentImage();
		if (img != null)
			return LocRenderWindow.getWindow(img);
		return null;
	}

	void runLocalization() {
		if (currentImage == null) {
			IJ.showMessage("No data selected. Select image and press \"Use active image\"");
			return;
		}
		
		if(cpp ==null || processingThread!=null)
			return;
		
		ImagePlus srcimg=currentImage;

		float pixelsizeX = 0.001f * (int)spinnerPixelSizeX.getValue();
		float pixelsizeY = 0.001f * (int)spinnerPixelSizeY.getValue();
		
		int zoom = Integer.parseInt(spinnerZoom.getValue().toString());
		
		datasetName = srcimg.getTitle();
		
		//ImgPlus<UnsignedShortType> image = (ImgPlus<UnsignedShortType>)currentData.getImgPlus();
		//ImagePlus im2 = ImageJFunctions.wrapUnsignedShort(image, "");
		ImageStack stack = srcimg.getImageStack();

		if(!(stack.getPixels(1) instanceof short[])) {
			IJ.showMessage("Only 16-bit image stacks are supported");
			return;
		}

		MainUI _this=this;

		ICameraCalibration cameraCalib = getCameraCalibration();
		
		Localizer loc;
		try {
			loc = createLocalizer(stack.getWidth(),stack.getHeight());
		} 
		catch(RuntimeException e) {
			cameraCalib.destroy();
			
			IJ.showMessage(e.getMessage());
			return;
		}

		Runnable r = new Runnable() {
			public void run() {
				if(!srcimg.lock())
					return;
		
				LocResults data = loc.localization(stack, cameraCalib, new Vector3( pixelsizeX, pixelsizeY, 1.0f ));
				cameraCalib.destroy();
				
				srcimg.unlock();
		
				if (data != null) {
					SwingUtilities.invokeLater(new Runnable() {
						@Override
						public void run() {
							LocRenderWindow wnd = new LocRenderWindow(data, datasetName, _this, ui, null, null, cpp, srcimg.getWidth()*zoom, srcimg.getHeight()*zoom);
							updateRender(wnd);
						}
					});
				}

				processingThread=null;
			};
		};
		
		processingThread=new Thread(r);
		processingThread.start();
		
	}
	
	@Override
	public void dispose()
	{
		if (cpp != null) {
			cpp.Close();
			cpp=null;
		}
		
		setVisible(false);
	}


	public void setUi(UIService ui) {
		this.ui=ui;
	}

	@Override
	public void showInfo(String info) {
		lblInfo.setText(info);
	}
}
