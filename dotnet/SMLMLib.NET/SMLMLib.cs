using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace SMLMLib
{
    public class Context : IDisposable
    {
        IntPtr instance;


        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr Context_Create();

        [SuppressUnmanagedCodeSecurity]
        [DllImport(SMLMLib.DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Context_Destroy(IntPtr ctx);


        public IntPtr Instance
        {
            get { return instance; }
        }

        public Context()
        {
            instance = Context_Create();
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                if (instance != IntPtr.Zero)
                {
                    Context_Destroy(instance);
                    instance = IntPtr.Zero;
                }

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~Context()
        {
           // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
           Dispose(false);
        }

        // This code added to correctly implement the disposable pattern.
        void IDisposable.Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            GC.SuppressFinalize(this);
        }
        #endregion

    }

    public unsafe class SMLMLib
    {
        public const string DllName = "photonpy.dll";



        
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);

        public static void SelectNativeLibrary(string baseDir, bool debugMode)
        {
            string dirname = debugMode ? "Debug" : "Release";
            string finalDir = baseDir + Path.DirectorySeparatorChar + "photonpy/x64" + 
                Path.DirectorySeparatorChar + dirname + Path.DirectorySeparatorChar;

            Trace.WriteLine("Selected native dll directory: " + finalDir);
            SetDllDirectory(finalDir);
        }

    }
}
