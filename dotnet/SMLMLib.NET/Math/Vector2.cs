using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace SMLMLib
{
	[System.ComponentModel.TypeConverter(typeof(StructTypeConverter<Vector2>))]
	public struct Vector2
	{	
		[XmlAttribute] public float x;
		[XmlAttribute] public float y;
		
		[XmlIgnore] public float X { get { return x; } set { x=value; }}
		[XmlIgnore] public float Y { get { return y; } set { y=value; }}
		
		public Vector2(float X,float Y)
		{ x = X; y = Y; }

		public Vector2(int X, int Y)
		{ x = X; y = Y; }

		#region Operators

		public static Vector2 operator*(Vector2 v, float f)
		{
			Vector2 r;
			r.x = v.x * f;
			r.y = v.y * f;
			return r;
		}

		public static Vector2 operator*(Vector2 a, Vector2 b)
		{
			Vector2 r;
			r.x = a.x * b.x;
			r.y = a.y * b.y;
			return r;
		}

		public static Vector2 operator*(float f, Vector2 v)
		{
			Vector2 r;
			r.x = v.x * f;
			r.y = v.y * f;
			return r;
		}

		public static Vector2 operator /(Vector2 v, float f)
		{
			Vector2 r;
			r.x = v.x / f;
			r.y = v.y / f;
			return r;
		}

		public static Vector2 operator +(Vector2 a, Vector2 b)
		{
			Vector2 r;
			r.x = a.x + b.x;
			r.y = a.y + b.y;
			return r;
		}

		public static Vector2 operator -(Vector2 a, Vector2 b)
		{
			Vector2 r;
			r.x = a.x - b.x;
			r.y = a.y - b.y;
			return r;
		}

		public static Vector2 operator -(Vector2 a)
		{
			return new Vector2(-a.x, -a.y);
		}

		// Dot-product
		public float Dot(Vector2 a)
		{
			return a.x * x + a.y * y;
		}

		#endregion

		public float Length
		{
			get { return (float)System.Math.Sqrt((double)(x * x + y * y)); }
		}

		public float SqLength
		{
			get { return x * x + y * y; }
		}
		
		public Vector2 Multiply (Vector2 v)
		{
			return new Vector2(v.x * x, v.y * y);
		}

		public Vector2 Ortho
		{
			get { return new Vector2(-y, x); }
		}

		public Vector2 Normalized
		{
			get
			{
				Vector2 v = this;
				v.Normalize();
				return v;
			}
		}

		/// <summary>
		/// Normalizes the vector
		/// </summary>
		/// <returns>Original length</returns>
		public float Normalize()
		{
			float len = Length;
			if (len < 0.0001f)
				return 0.0f;

			float invLength = 1.0f / len;
			x *= invLength;
			y *= invLength;
			return len;
		}

		// Calculates the exact nearest point, not just one of the box'es vertices
		public static Vector2 NearestBoxPoint(Vector2 min, Vector2 max, Vector2 pos)
		{
			Vector2 ret;
			Vector2 mid = (max + min) * 0.5f;
			if(pos.x < min.x) ret.x = min.x;
			else if(pos.x > max.x) ret.x = max.x;
			else ret.x = pos.x;

			if(pos.y < min.y) ret.y = min.y;
			else if(pos.y > max.y) ret.y = max.y;
			else ret.y = pos.y;
			return ret;
		}

		public override string ToString()
		{
			return String.Format("X={0}, Y={1}", x, y);
		}

		public Vector2 Absolute()
		{
			return new Vector2(Math.Abs(x), Math.Abs(y));
		}

		public float DistanceToRay(Vector2 a, Vector2 b)
		{
			Vector2 o = (b - a).Ortho;
			o.Normalize();

			float d = o.Dot(a);
			float dist = o.Dot(this);

			return Math.Abs(dist - d);
		}

		public float DistanceToLine(Vector2 a, Vector2 b)
		{
			Vector2 d = b - a;
			float sa = d.Dot(a);
			float sb = d.Dot(b);
			float st = d.Dot(this);

			bool useRay;

			if (sb < sa)
				useRay = st > sb && st < sa;
			else
				useRay = st > sa && st < sb;

			if (useRay)
				return DistanceToRay(a, b);

			return Math.Min((this - a).Length, (this - b).Length);
		}

		public Vector2 ClosestPosOnLine(Vector2 a, Vector2 b)
		{
			Vector2 diff = b - a;
			float sa = diff.Dot(a);
			float sb = diff.Dot(b);
			float st = diff.Dot(this);

			bool useRay;

			if (sb < sa)
				useRay = st > sb && st < sa;
			else
				useRay = st > sa && st < sb;

			if (useRay) {
				Vector2 o = (b - a).Ortho;
				o.Normalize();

				float d = o.Dot(a);
				float dist = o.Dot(this);

				// Project vector on line
				return this + (d-dist) * o;
			}

			float distToA = (this - a).Length;
			float distTob = (this - b).Length;

			if (distToA < distTob)
				return a;
			return b;
		}


		public static Vector2 FromAngle(float angle)
		{
			return new Vector2((float)Math.Cos(angle), (float)Math.Sin(angle));
		}

		public Vector2 SnapToGrid(float gridSize)
		{
			float _x = gridSize * (float)Math.Round(x / gridSize);
			float _y = gridSize * (float)Math.Round(y / gridSize);
			return new Vector2(_x, _y);
		}

		public void Min(Vector2 o)
		{
			if (o.x < x) x = o.x;
			if (o.y < y) y = o.y;
		}

		public void Max(Vector2 o)
		{
			if (o.x > x) x = o.x;
			if (o.y > y) y = o.y;
		}
	}
}
