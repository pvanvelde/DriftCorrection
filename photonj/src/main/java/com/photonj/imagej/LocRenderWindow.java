package com.photonj.imagej;


import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.HashMap;

import org.scijava.ui.UIService;

import com.photonj.binding.Binding;
import com.photonj.binding.Vector3;

import ij.ImageListener;
import ij.ImagePlus;
import ij.gui.ImageCanvas;
import ij.gui.ImageWindow;
import ij.gui.Line;
import ij.gui.Roi;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;


public class LocRenderWindow implements WindowListener, MouseListener, MouseMotionListener, KeyListener {

	LocResults data;
	ImagePlus image;
	private UIService ui;
	private String name;
	private Binding binding;
	
	View view;
	int renderWidth, renderHeight;

	public static class View {
		public Transform transform;
		public float minZ, maxZ, sigma;
		
		public View(Transform transform, float minZ, float maxZ, float sigma) {
			this.transform = transform;
			this.minZ = minZ;
			this.maxZ = maxZ;
			this.sigma = sigma;
		}
		
		public boolean equals(View v) {
			return v.sigma == sigma && v.minZ == minZ && v.maxZ == maxZ &&
					v.transform.equals(transform);
		}

		@Override
		public String toString() {
			return String.format("View{Transform:%s, Sigma=%f}", transform.toString(), sigma);
		}
	}

	LocRenderWindow parent;
	Roi zoomROI; // the ROI in the parent window that is linked to this view
	ArrayList<LocRenderWindow> subViews=new ArrayList<LocRenderWindow>();

	static HashMap<ImagePlus, LocRenderWindow> renderWindows=new HashMap<>();
	
	public interface ILocInfoDisplay {
		void showInfo(String info);
	}
	
	ILocInfoDisplay infoDisplay;
	
	public LocRenderWindow(LocResults data, String name, ILocInfoDisplay infoDisplay, UIService ui, Roi zoomRoi, LocRenderWindow parent, Binding binding, int renderWidth, int renderHeight) {
		this.data=data;
		this.name =name;
		this.ui=ui;
		this.binding=binding;
		this.infoDisplay = infoDisplay;
		this.zoomROI = zoomRoi;
		this.parent = parent;
		this.renderWidth = renderWidth;
		this.renderHeight = renderHeight;
		
		float minZ=0.0f, maxZ=0.0f;
		
		if (!data.results.isEmpty()) {
			minZ = data.kdTree.boundingSphere.pos.z - data.kdTree.boundingSphere.radius;
			maxZ = data.kdTree.boundingSphere.pos.z + data.kdTree.boundingSphere.radius;
		}
		
		view = new View(new Transform(), minZ, maxZ, 1.0f);
	}

	public void update(View view) {
		this.view = view;
		
		// TODO: Use minz,maxz from view
		RandomAccessibleInterval<FloatType> img = data.renderToImage(renderWidth, renderHeight, view.transform, view.sigma, binding);
		ImagePlus ip = ImageJFunctions.wrap(img, name + " render");
		float[] pixels = (float[])ip.getStack().getPixels(1);
		//System.out.println("#Pixels" + pixels.length);
		float maxVal = 0.0f;
		for(int i=0;i<pixels.length;i++)
			if (pixels[i]>maxVal) maxVal = pixels[i];
		
		if(image==null) {
			image = ip;
			image.setDisplayRange(0.0f, maxVal);
			
			// This is pretty ugly but I can't find another way to access the image canvas/window after running ui.show()
			ImagePlus.addImageListener(new ImageListener() {
				@Override
				public void imageUpdated(ImagePlus imp) {
				}
				
				@Override
				public void imageOpened(ImagePlus imp) {
					if (imp == image) {
						ImagePlus.removeImageListener(this);
						registerListeners(image);
					}
				}
				
				@Override
				public void imageClosed(ImagePlus imp) {
				}
			});
			ui.show(image);
			
		} else {
			double min = image.getDisplayRangeMin();
			double max = image.getDisplayRangeMin();
				
			image.setStack(ip.getImageStack());
			image.setDisplayRange(min, max);
		}

	}
	
	void registerListeners(ImagePlus image)  {
		
		ImageCanvas canvas = image.getCanvas();
		ImageWindow wnd = image.getWindow();
		
		renderWindows.put(image, this);
					
		wnd.addWindowListener(this);
		canvas.addMouseMotionListener(this);
		//canvas.addMouseListener(this);
		canvas.addKeyListener(this);
	}
	
	@Override
	public void mouseDragged(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}
	
	Vector3 pixelToLocSpace(Point2D.Double pt) {
		return view.transform.inverse().transform(new Vector3((float)pt.x,(float)pt.y,0.0f));
	}

	@Override
	public void mouseMoved(MouseEvent e) {
		// TODO Auto-generated method stub
		
		if (infoDisplay != null) {
			ImageCanvas canvas=image.getCanvas();
			
			double x = canvas.offScreenXD(e.getX());
			double y = canvas.offScreenYD(e.getY());

			Vector3 tp = pixelToLocSpace(new Point2D.Double(x,y));
						
			infoDisplay.showInfo(String.format("Mouse x=%.2f, y=%.2f, %d spots fitted",tp.x,tp.y, data.results.size()));
		}
		
	}

	@Override
	public void mouseClicked(MouseEvent e) {
		
	}

	@Override
	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void windowOpened(WindowEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void windowClosing(WindowEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void windowClosed(WindowEvent e) {
		// TODO Auto-generated method stub
		
		renderWindows.remove(image);
		
		System.out.println("Closing render window");
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
	public void windowActivated(WindowEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void windowDeactivated(WindowEvent e) {
		// TODO Auto-generated method stub
		
	}
	

	public static LocRenderWindow getWindow(ImagePlus img) {
		return renderWindows.get(img);
	}

	public LocResults getResults() {
		return data;
	}

	public static void imageClosed(ImagePlus imp) {

		LocRenderWindow wnd = getWindow(imp);
		
		if (wnd != null) {
			wnd.close();
		}
		
	}

	private void close() {
		
		renderWindows.remove(image);
		
	}

	@Override
	public void keyTyped(KeyEvent e) {

		
	}

	@Override
	public void keyPressed(KeyEvent e) {
		System.out.println(KeyEvent.getKeyText(e.getKeyCode()));
	}

	@Override
	public void keyReleased(KeyEvent e) {
	}


	View getZoomView(Roi roi) {
		
		if (data.empty()) 
			return null;
		
		if (roi.getType() == Roi.RECTANGLE) {
			Rectangle2D.Double rc = roi.getFloatBounds();
			Vector3 origin = pixelToLocSpace(new Point2D.Double(rc.getMinX(), rc.getMinY()));
	
			float zoomX = image.getWidth() / (float)rc.getWidth();
			float zoomY = image.getHeight() / (float)rc.getHeight();
			float zoom = Math.min(zoomX, zoomY);
			
			Vector3 newX = view.transform.m[0].mul(zoom);
			Vector3 newY = view.transform.m[1].mul(zoom);

			Transform transform = new Transform(new Vector3[] { newX,newY,view.transform.m[2] }, new Vector3());
			transform.t = transform.transform(origin).neg();
			//System.out.println("new offset:"+transform.t.toString());
			return new View(transform, view.minZ, view.maxZ, view.sigma/zoom);
		}
		else if(roi.getType() == Roi.LINE) {
			Line line = (Line)roi;
	    
			//new LocRenderWindow(data, name, ui, binding)
			
			// compute line start and end positions in dataset-space
			Vector3 startPos = pixelToLocSpace(new Point2D.Double(line.x1d, line.y1d));
			Vector3 endPos = pixelToLocSpace(new Point2D.Double(line.x2d, line.y2d));

			// now build the matrix that converts from subview-space to dataset-space
			// this is build up from the parent subview-to-dataset matrix and the user selected line
			Vector3 imgXAxis = endPos.minus(startPos).normalized();
			Vector3 imgYAxis = view.transform.inverse().m[2].normalized().neg(); // the new Y axis is the axis that is going into the screen in the parent view
			Vector3 imgZAxis = imgXAxis.crossProduct(imgYAxis);
			
			// we need the inverse matrix of that
			Vector3[] invM = new Vector3[] {imgXAxis, imgYAxis, imgZAxis };
			Vector3[] M = Vector3.matrixInverse(invM);

			float scale = image.getWidth() / endPos.minus(startPos).len();
			Transform transform = new Transform(Vector3.mul(M,scale), new Vector3());
			transform.t = transform.transform(startPos).neg();

			// scale the view so start to end fit in X
			Vector3[] extents = data.computeExtent(transform);
			transform.t.y = image.getHeight() * 0.5f - 0.5f * (extents[1].y+extents[0].y);

			float w = line.getStrokeWidth();
			//System.out.println("Line:"+lineLength + " width:"+w);

			return new View(transform, -1000.0f, 1000.0f, view.sigma);
			
		}
		return null;
	}


	public void renderMarkedROI() {

		double lineLength = 0;
		Roi roi = image.getRoi();
		
		View subview = getZoomView(roi);
		
		if (subview != null) {
			LocRenderWindow subViewWindow = new LocRenderWindow(data, name, infoDisplay, this.ui, roi, this, binding, image.getWidth(), image.getHeight());
			subViewWindow.update(subview);
			subViews.add(subViewWindow);
		}
		
	}

	public static void timerUpdate() {
		for (ImagePlus img : renderWindows.keySet()) {
			LocRenderWindow wnd = renderWindows.get(img);
			
			if (wnd.zoomROI != null && wnd.zoomROI.isVisible()) {
				View view = wnd.parent.getZoomView(wnd.zoomROI);
				
				if (!view.equals(wnd.view)) {
					
					//System.out.println("updating view: " + view.toString());
					wnd.update(view);
				}
			}
			
		}
	}

}

