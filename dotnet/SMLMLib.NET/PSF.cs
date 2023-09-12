using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace SMLMLib
{
    public class PSF : IDisposable
    {
        IntPtr instance;
		int[] sampleDims;
		int numConst, numParams, numDiag, sampleCount;

		public int SampleCount { get => sampleCount; }
		public int NumConst { get => numConst; }
		public int NumParams { get => numParams; }
		public int NumDiag { get => numDiag; }

		internal PSF(IntPtr inst)
        {
            this.instance = inst;

			int dims = PSF_SampleIndexDims(inst);
			sampleDims = new int[dims];

			for (int i=0;i<dims;i++)
				sampleDims[i] = PSF_SampleSize(inst, i);

			numConst = PSF_NumConstants(inst);
			numDiag = PSF_NumDiag(inst);
			numParams = PSF_NumParams(inst);
			sampleCount = PSF_SampleCount(inst);
        }

		public float[,] ExpectedValue(float[,] params_, float[,] constants=null, int[,] spotpos=null)
		{
			int numspots = params_.GetLength(0);
			float[,] ev = new float[numspots, sampleCount];

			PSF_ComputeExpectedValue(instance, numspots, params_, constants, spotpos, ev);
			return ev;
		}


        void IDisposable.Dispose()
        {
            if (instance != IntPtr.Zero)
                PSF_Delete(instance);
        }




		/*
		 // C/Python API - All pointers are host memory
	CDLL_EXPORT void PSF_Delete(PSF* psf);
		CDLL_EXPORT const char* PSF_ThetaFormat(PSF * psf);
		CDLL_EXPORT int PSF_ThetaSize(PSF* psf);
		CDLL_EXPORT int PSF_SampleCount(PSF* psf);
		CDLL_EXPORT int PSF_NumConstants(PSF* psf);
		CDLL_EXPORT int PSF_NumDiag(PSF* psf);
		CDLL_EXPORT void PSF_ComputeExpectedValue(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* ev);
		//CDLL_EXPORT void PSF_ComputeInitialEstimate(PSF* psf, int numspots, const float* sample, const float* constants, float* theta);
		CDLL_EXPORT void PSF_ComputeMLE(PSF* psf, int numspots, const float* sample, const float* constants, const int* spotpos, const float* initial, float* theta,
			float* diagnostics, int* iterations, float* trace, int traceBufLen);
		CDLL_EXPORT void PSF_ComputeFisherMatrix(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* fi);
		CDLL_EXPORT void PSF_ComputeDerivatives(PSF* psf, int numspots, const float* theta, const float* constants, const int* spotpos, float* derivatives, float* ev);
		CDLL_EXPORT int PSF_SampleIndexDims(PSF* psf);
CDLL_EXPORT int PSF_SampleSize(PSF* psf, int dim);

		*/

		#region Native API

		[SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void PSF_Delete(IntPtr psfInstance);


        //IntPtr ptr = foo();
        //string str = Marshal.PtrToStringAuto(ptr);
        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "PSF_InputFormat")]
        internal static extern IntPtr PSF_ParamFormat(IntPtr psfInstance);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PSF_NumParams(IntPtr psfInstance);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PSF_SampleCount(IntPtr psfInstance);

		[SuppressUnmanagedCodeSecurity]
		[DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
		internal static extern int PSF_SampleIndexDims(IntPtr psf);

		[SuppressUnmanagedCodeSecurity]
		[DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
		internal static extern int PSF_SampleSize(IntPtr psf, int dim);


		[SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PSF_NumConstants(IntPtr psfInstance);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PSF_NumDiag(IntPtr psfInstance);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void PSF_ComputeExpectedValue(IntPtr psf, int numspots, 
			float[,] theta, [In] float[,] constants, [In] int[,] spotpos, float[,] ev);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void PSF_Estimate(IntPtr psfInstance, int numspots, [In] float[] sample, [In] float[] constants, [In] int[] spotpos, [In] float[] initial, [Out] float[] _params,
            [Out] float[] diagnostics, [Out] int[] iterations, [Out] float[] trace, int traceBufLen);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void PSF_ComputeFisherMatrix(IntPtr psfInstance, int numspots, [In] float[] theta, [In] float[] constants, [In] int[] spotpos, [Out] float[] fi);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void PSF_ComputeDerivatives(IntPtr psf, int numspots, [In] float[] params_, [In] float[] constants, [In] int[] spotpos, [Out] float[] derivatives, [Out] float[] ev);


        #endregion
    }
}
