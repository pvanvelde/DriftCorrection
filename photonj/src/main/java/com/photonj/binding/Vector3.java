package com.photonj.binding;


public class Vector3 {
	
	public float x,y,z;

	public Vector3(float x,float y,float z) {
		this.x=x;
		this.y=y;
		this.z=z;
	}
	
	public Vector3() {
		
	}
	
	public void set(Vector3 v) {
		x=v.x;y=v.y;z=v.z;
	}
	
	public Vector3 div(float a) {
		return mul(1.0f/a);
	}
	
	public Vector3 mul(float a) {
		return new Vector3(x*a,y*a,z*a);
	}
	
	public Vector3 add(Vector3 a) {
		return new Vector3(a.x+x,a.y+y,a.z+z);
	}
	
	public float dot(Vector3 v) {
		return x*v.x+y*v.y+z*v.z;
	}
	
	public Vector3 mul(Vector3 v) {
		return new Vector3(x*v.x,y*v.y,z*v.z);
	}
	
	public float get(int axis) {
		switch(axis) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		}
		throw new RuntimeException("Vector3.get() invalid axis" + axis);
	}
	
	public void set(int axis, float v) {
		switch(axis) {
		case 0: x=v; return;
		case 1: y=v; return;
		case 2: z=v; return;
		}
		throw new RuntimeException("Vector3.set() invalid axis" + axis);
	}
	

	public float len() {
		return (float)Math.sqrt(dot(this));
	}
	
	public float[] toArray() { 
		return new float[] { x,y,z};
	}
	
	public static float[][] toArrays(Vector3[] rows) {
		float[][] r = new float[rows.length][];
		for(int i=0;i<rows.length;i++)
			r[i]=rows[i].toArray();
		return r;
	}
	
	public static float matrixDet(Vector3[] rows) {
		float[][] m = toArrays(rows);
		
		// computes the inverse of a matrix m
		return m[0] [0] * (m[1][ 1] * m[2][2] - m[2][ 1] * m[1][2]) -
		             m[0] [1] * (m[1][ 0] * m[2][2] - m[1][ 2] * m[2][0]) +
		             m[0] [2] * (m[1][ 0] * m[2][1] - m[1][ 1] * m[2][0]);
	}
	
	public static Vector3[] matrixInverse(Vector3[] xyz) {

		float invdet = 1 / matrixDet(xyz);
		
		float[][] m = toArrays(xyz);

		Vector3[] inv=new Vector3[3];
		inv[0] = new Vector3(
				(m[1][ 1] * m[2][ 2] - m[2][1] * m[1][ 2]) * invdet,
				(m[0][ 2] * m[2][ 1] - m[0][1] * m[2][ 2]) * invdet,
				(m[0][ 1] * m[1][ 2] - m[0][2] * m[1][ 1]) * invdet);

		inv[1] = new Vector3( 
				(m[1][ 2] * m[2][ 0] - m[1][0] * m[2][ 2]) * invdet,
				(m[0][ 0] * m[2][ 2] - m[0][2] * m[2][ 0]) * invdet,
				(m[1][ 0] * m[0][ 2] - m[0][0] * m[1][ 2]) * invdet);
		
		inv[2] = new Vector3(
				(m[1][ 0] * m[2][ 1] - m[2][0] * m[1][ 1]) * invdet,
				(m[2][ 0] * m[0][ 1] - m[0][0] * m[2][ 1]) * invdet,
				(m[0][ 0] * m[1][ 1] - m[1][0] * m[0][ 1]) * invdet);
		
		return inv;
	}

	public static Vector3 matrixApply(Vector3[] m, Vector3 pos) {
		return new Vector3( m[0].dot(pos), m[1].dot(pos), m[2].dot(pos));
	}

	public void setLength(float l) {
		set(mul(l/len()));
	}
	
	public Vector3 normalized() {
		return div(len());
	}
 
	public Vector3 neg() {
		return new Vector3(-x,-y,-z);
	}
	
	@Override
	public String toString() {
		return String.format("%.3f, %.3f, %.3f", x,y,z);
	}

	public Vector3 minus(Vector3 v) {
		return new Vector3(x-v.x,y-v.y,z-v.z);
	}

	public Vector3 crossProduct(Vector3 b) {
		return new Vector3(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
	}

	public static Vector3 min(Vector3[] pos) {
		Vector3 m = pos[0].mul(1.0f); // make copy
		for (int i=1;i<pos.length;i++) {
			Vector3 v = pos[i];
			if (m.x > v.x) m.x=v.x;
			if (m.y > v.y) m.y=v.y;
			if (m.z > v.z) m.z=v.z; 
		}
		return m;
	}

	public static Vector3 max(Vector3[] pos) {
		Vector3 m = pos[0].mul(1.0f); // make copy
		for (int i=1;i<pos.length;i++) {
			Vector3 v = pos[i];
			if (m.x < v.x) m.x=v.x;
			if (m.y < v.y) m.y=v.y;
			if (m.z < v.z) m.z=v.z; 
		}
		return m;
	}

	public static Vector3[] mul(Vector3[] m, float a) {
		Vector3[] r = new Vector3[m.length];
		for(int i=0;i<m.length;i++)
			r[i]=m[i].mul(a);
		return r;
	}
	
	public boolean equals(Vector3 v) {
		return x==v.x && y==v.y && z==v.z;
	}
}
