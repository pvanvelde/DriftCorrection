package com.photonj.imagej;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.*;
import org.apache.commons.math3.stat.descriptive.rank.Median;

import com.photonj.binding.IHasLocation;
import com.photonj.binding.Vector3;

public class KDTree<T extends IHasLocation> {
	IHasLocation[] data;
	Sphere boundingSphere;
	
	public static class Sphere {
		public Vector3 pos=new Vector3();
		public float radius;
	}
	
	// These are only valid if data=null
	KDTree[] nodes = new KDTree[2];
	float value;
	int axis;
	
	public boolean isLeafNode() {
		return data==null;
	}
	
	
	static float[][] getCoords(List<IHasLocation> items) {
		float[][] r = new float[3][items.size()];
		int idx=0;
		for (IHasLocation i : items) {
			Vector3 pos = i.getLocation();
			for(int ax=0;ax<3;ax++)
				r[ax][idx] = pos.get(ax);
			idx++;
		}
		return r;
	}
	
	static float[] computeStats(float[] items) {
		float Ex=0.0f,Ex2=0.0f;
		float min = items[0];
		float max = items[0];
		for (float s : items) {
			Ex += s-min;
			Ex2 += (s-min)*(s-min);
			if (s < min) min = s;
			if (s > max) max = s;
		}
		int n = items.length;
		float variance = (Ex2-(Ex*Ex)/n)/n;
		float mean = Ex/n;
	    return new float[] {variance,mean, min, max};
	}
	
	public KDTree(List<IHasLocation> items, int maxNodeSize) {
		float[][] data=getCoords(items);
		float[][] stats = new float[3][];
		
		for (int i=0;i<3;i++)
			stats[i]=computeStats(data[i]);
		
		if (items.size()>maxNodeSize) {
			int bestAxis=0;
			float bestVar=stats[0][0];
			for (int i=1;i<3;i++) {
				float var = stats[i][0];
				if (var > bestVar) {
					bestVar = var;
					bestAxis=i;
				}
			}
	
			double[] v=new double[items.size()];
			for(int i=0;i<v.length;i++)
				v[i]=data[bestAxis][i];
			
			float median = (float)new Median().evaluate(v);
			
			value = median;
			axis = bestAxis;
			
			ArrayList<IHasLocation> l0 = new ArrayList<IHasLocation>();
			ArrayList<IHasLocation> l1 = new ArrayList<IHasLocation>();
			int idx=0;
			for (IHasLocation i : items) {
				if (v[idx] >= value) {
					l1.add(i);
				}
				else {
					l0.add(i);
				}
				idx++;
			}
			
			// some numerical instability can cause this
			// just put everything in one node
			if (!l0.isEmpty() && !l1.isEmpty()) {
				nodes=new KDTree[2];
				nodes[0]=new KDTree<T>(l0, maxNodeSize);
				nodes[1]=new KDTree<T>(l1, maxNodeSize);
			}
			else {
				System.out.println(String.format("KDTree: Creating leaf node with %d spot.", items.size()));
			}
		}

		if (nodes == null) { // leaf node
			this.data=new IHasLocation[items.size()];
			for(int i=0;i<data.length;i++)
				this.data[i]=items.get(i);
		}
		
		boundingSphere = new Sphere();
		for (int i=0;i<3;i++) {
			float min=stats[i][2], max=stats[i][3];
			float center = (max+min)*0.5f;
			boundingSphere.pos.set(i, center);
			if (boundingSphere.radius < max-center)
				boundingSphere.radius = max-center;
		}
	}

}


